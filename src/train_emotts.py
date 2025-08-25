import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import soundfile as sf
from scipy.signal import resample_poly
import math
from transformers import (
    SpeechT5Processor, SpeechT5ForTextToSpeech,
    AutoFeatureExtractor, WavLMModel,
    Seq2SeqTrainingArguments, Seq2SeqTrainer)
from speechbrain.pretrained import EncoderClassifier
from datasets import Value
from src.utils import read_manifest_csv, build_hf_dataset
from src.style_modules import StyleAdaptor, StyleSpeakerFusion, LossWeights

# Optional LoRA / PEFT
from peft import LoraConfig, get_peft_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Manifest CSV with path,text,emotion,speaker,style_path",default="../data/ravdess_manifest.csv")
    ap.add_argument("--out_dir", default="runs/emotts_ravdess")
    ap.add_argument("--base_tts", default="microsoft/speecht5_tts")
    ap.add_argument("--vocoder", default="microsoft/speecht5_hifigan")
    ap.add_argument("--ssl_name", default="microsoft/wavlm-base-plus")
    ap.add_argument("--spk_embedder", default="speechbrain/spkrec-ecapa-voxceleb")

    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--emo_ce_weight", type=float, default=0.2)
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    # LoRA / PEFT
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target", choices=["attn", "mlp", "all"], default="attn",
                    help="Which submodules to wrap with LoRA adapters")
    ap.add_argument("--merge_lora_on_save", action="store_true", default=True)
    return ap.parse_args()


def find_target_linear_names(model, want):
    attn_keys = ["q", "k", "v", "o", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_keys = ["fc1", "fc2", "wi", "wo", "down", "up", "proj"]
    keys = set()
    if want in ("attn", "all"):
        keys.update(attn_keys)
    if want in ("mlp", "all"):
        keys.update(mlp_keys)

    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lname = name.split(".")[-1].lower()
            if any(k in lname for k in keys):
                found.add(lname)
    if not found:
        found = {"q", "k", "v", "o", "wi", "wo"}
    return sorted(found)


def build_emotion_mapping(ds_dict):
    # Collect unique emotion strings across splits
    emos = set(ds_dict["train"]["emotion"])
    emos.update(ds_dict["test"]["emotion"])
    emos = sorted(emos)
    emo2id = {e: i for i, e in enumerate(emos)}
    print(f"[Info] Detected emotions ({len(emos)}): {emos}")
    return emos, emo2id


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load rows + build HF dataset (decodes audio + style_audio @ 16k)
    rows = read_manifest_csv(args.csv)
    ds = build_hf_dataset(rows, sampling_rate=16000)

    ds = ds.cast_column("audio", Value("string"))
    ds = ds.cast_column("style_audio", Value("string"))
    ds = ds.train_test_split(test_size=0.1, seed=42)

    # 2) Build emotion mapping from the actual CSV and (re)encode emotion_id
    emotions, emo2id = build_emotion_mapping(ds)

    def remap_emo(example):
        example["emotion_id"] = emo2id[example["emotion"]]
        return example

    for split in ["train", "test"]:
        ds[split] = ds[split].map(remap_emo, desc=f"Remapping emotions ({split})")

    # 3) Load processors / models
    processor = SpeechT5Processor.from_pretrained(args.base_tts)

    wavlm_fe = AutoFeatureExtractor.from_pretrained(args.ssl_name)
    wavlm_model = WavLMModel.from_pretrained(args.ssl_name).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wavlm_model.to(device)

    spk_model = EncoderClassifier.from_hparams(
        source=args.spk_embedder,
        run_opts={"device": device},
        savedir=os.path.join(args.out_dir, "spkrec_cache"),
    )

    # Determine speaker embedding dim dynamically (works for ECAPA/x-vector, etc.)
    with torch.no_grad():
        dummy = torch.zeros(1, 16000).to(device)  # 1 second of silence @ 16k
        spk_probe = spk_model.encode_batch(dummy)
        spk_dim = spk_probe.shape[-1]
    print(f"[Info] Detected speaker embedding dim: {spk_dim}")

    def read_mono_16k(path: str):
        """Load a wav path -> mono float32 @ 16 kHz."""
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype("float32", copy=False)
        if sr != 16000:
            g = math.gcd(sr, 16000)
            up, down = 16000 // g, sr // g
            wav = resample_poly(wav, up, down).astype("float32", copy=False)
            sr = 16000
        return wav, sr

    # 4) Example preparation
    def prepare_example(example):
        # --- load training audio (mono, 16 kHz) ---
        wav_arr, wav_sr = read_mono_16k(example["audio"])  # example["audio"] is a string path now

        # Text -> labels (mel) via SpeechT5 processor
        ex = processor(
            text=example["text"],
            audio_target=wav_arr,
            sampling_rate=wav_sr,
            return_attention_mask=False,
        )
        ex["labels"] = ex["labels"][0]
        # ex["stop_labels"] = ex["stop_labels"][0]

        # Speaker x-vector from the same clip
        with torch.no_grad():
            wav_tensor = torch.tensor(wav_arr, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T]
            spk_emb = spk_model.encode_batch(wav_tensor)  # [1, 1, D]
            spk_emb = torch.nn.functional.normalize(spk_emb, dim=2).squeeze(0).squeeze(0).cpu().numpy()
        ex["speaker_embeddings"] = spk_emb  # [D]

        # --- load style/reference audio and build WavLM style vector ---
        style_arr, style_sr = read_mono_16k(example["style_audio"])
        inputs = wavlm_fe(style_arr, sampling_rate=style_sr, return_tensors="pt")
        with torch.no_grad():
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            hid = wavlm_model(**inputs).last_hidden_state  # [1, T, 768]
            style_vec = hid.mean(dim=1).squeeze(0).cpu().numpy()
        ex["style_vec"] = style_vec

        ex["emotion_id"] = example["emotion_id"]  # from the earlier mapping step
        return ex

    ds_proc = {}
    for split in ["train", "test"]:
        ds_proc[split] = ds[split].map(
            prepare_example,
            remove_columns=ds[split].column_names,
            desc=f"Preparing {split}",
        )

    # 5) Collator
    @dataclass
    class TTSDataCollatorWithStyle:
        processor: Any
        model: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # 1) Pad text and label spectrograms
            input_ids = [{"input_ids": f["input_ids"]} for f in features]
            label_features = [{"input_values": f["labels"]} for f in features]
            batch = self.processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

            # 2) Collect original target lengths (before padding)
            target_lengths = torch.tensor([len(f["input_values"]) for f in label_features], dtype=torch.long)

            # 3) Round lengths to multiple of reduction factor (if any), then trim labels to max_len
            rf = getattr(self.model.config, "reduction_factor", 1)
            if rf > 1:
                target_lengths = (target_lengths // rf) * rf
            max_len = int(target_lengths.max().item())

            # If the processor created a decoder mask, trim it to max_len; otherwise build one from lengths
            if "decoder_attention_mask" in batch:
                dec_mask = batch["decoder_attention_mask"][:, :max_len].to(torch.bool)
                del batch["decoder_attention_mask"]
            else:
                dec_mask = torch.arange(max_len)[None, :].lt(target_lengths[:, None])

            # Trim labels to max_len and mask padding with -100
            batch["labels"] = batch["labels"][:, :max_len]
            batch["labels"] = batch["labels"].masked_fill(~dec_mask.unsqueeze(-1), -100)

            # 4) Add extra features (speaker/style/emotion)
            batch["speaker_embeddings"] = torch.tensor([f["speaker_embeddings"] for f in features], dtype=torch.float32)
            batch["style_vec"] = torch.tensor([f["style_vec"] for f in features], dtype=torch.float32)
            batch["emotion_id"] = torch.tensor([f["emotion_id"] for f in features], dtype=torch.long)
            return batch

    # 6) Base TTS + optional LoRA
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(args.base_tts)
    tts_model.config.use_cache = False

    if args.use_lora:
        target_names = find_target_linear_names(tts_model, args.lora_target)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_names,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        tts_model = get_peft_model(tts_model, lora_cfg)
        tts_model.print_trainable_parameters()

    # Style adaptor (emotion head sized to detected classes)
    num_emotions = len(emotions)
    style_adaptor = StyleAdaptor(in_dim=768, latent_dim=256, num_emotions=num_emotions)
    fusion = StyleSpeakerFusion(spk_dim=spk_dim, style_dim=256, out_dim=512)
    loss_weights = LossWeights(emo_ce=args.emo_ce_weight)

    class EmoTTSTrainer(Seq2SeqTrainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.style_adaptor = style_adaptor.to(self.model.device)
            self.fusion = fusion.to(self.model.device)
            self.ce = nn.CrossEntropyLoss()

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            emotion_id = inputs.pop("emotion_id")
            style_vec = inputs.pop("style_vec")
            spk = inputs.pop("speaker_embeddings")

            style_latent, emo_logits = self.style_adaptor(style_vec)
            emo_loss = self.ce(emo_logits, emotion_id)

            fused = self.fusion(spk, style_latent)
            outputs = model(speaker_embeddings=fused, **inputs)

            loss = outputs.loss + loss_weights.emo_ce * emo_loss
            return (loss, outputs) if return_outputs else loss

    collator = TTSDataCollatorWithStyle(processor=processor, model=tts_model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=False,
        fp16=args.fp16,
        eval_strategy="steps",  # <-- correct HF arg
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.out_dir,'logs'),
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
        push_to_hub=args.push_to_hub,
        remove_unused_columns=False
    )


    try:
        trainer = EmoTTSTrainer(
            args=training_args,
            model=tts_model,
            train_dataset=ds_proc["train"],
            eval_dataset=ds_proc["test"],
            data_collator=collator,
            processing_class=processor,  # new API
        )
    except TypeError:
        trainer = EmoTTSTrainer(
            args=training_args,
            model=tts_model,
            train_dataset=ds_proc["train"],
            eval_dataset=ds_proc["test"],
            data_collator=collator,
            tokenizer=processor,  # old API
        )

    trainer.train()

    # Optionally merge LoRA adapters into base weights for simpler inference
    final_model = trainer.model
    if args.use_lora and args.merge_lora_on_save:
        try:
            final_model = final_model.merge_and_unload()
            print("Merged LoRA adapters into base weights.")
        except Exception as e:
            print("Warning: merge_and_unload failed; saving PEFT adapters instead.", e)

    final_model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    torch.save(style_adaptor.state_dict(), os.path.join(args.out_dir, "style_adaptor.pt"))
    torch.save(fusion.state_dict(), os.path.join(args.out_dir, "style_fusion.pt"))

    print("Training finished. Artifacts saved to:", args.out_dir)


if __name__ == "__main__":
    main()
