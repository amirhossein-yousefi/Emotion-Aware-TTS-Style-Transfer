import os
import argparse
import torch
import soundfile as sf

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, AutoProcessor, WavLMModel
from speechbrain.pretrained import EncoderClassifier
from transformers import AutoFeatureExtractor, WavLMModel
from style_modules import StyleAdaptor, StyleSpeakerFusion

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Directory with fine-tuned SpeechT5 & processor")
    ap.add_argument("--style_adaptor", default=None, help="Path to style_adaptor.pt (if not in ckpt_dir)")
    ap.add_argument("--style_fusion", default=None, help="Path to style_fusion.pt (if not in ckpt_dir)")

    ap.add_argument("--text", required=True)
    ap.add_argument("--style_wav", required=True, help="Reference utterance whose style/emotion to transfer")
    ap.add_argument("--target_spk_wav", default=None, help="Optional separate target-speaker wav; defaults to style_wav")

    ap.add_argument("--ssl_name", default="microsoft/wavlm-base-plus")
    ap.add_argument("--spk_embedder", default="speechbrain/spkrec-ecapa-voxceleb")
    ap.add_argument("--out_wav", default="emotts_out.wav")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base TTS + vocoder + processor
    processor = SpeechT5Processor.from_pretrained(args.ckpt_dir)
    tts = SpeechT5ForTextToSpeech.from_pretrained(args.ckpt_dir).to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()

    # Load style modules
    style_adaptor = StyleAdaptor()
    fusion = StyleSpeakerFusion()
    style_path = args.style_adaptor or os.path.join(args.ckpt_dir, "style_adaptor.pt")
    fusion_path = args.style_fusion or os.path.join(args.ckpt_dir, "style_fusion.pt")
    style_adaptor.load_state_dict(torch.load(style_path, map_location="cpu"))
    fusion.load_state_dict(torch.load(fusion_path, map_location="cpu"))
    style_adaptor, fusion = style_adaptor.to(device).eval(), fusion.to(device).eval()

    # Load WavLM + processor for style features
    ssl_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    ssl_model = WavLMModel.from_pretrained(args.ssl_name).to(device).eval()

    # Speaker embedder
    spk_model = EncoderClassifier.from_hparams(
        source=args.spk_embedder,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", args.spk_embedder)
    )

    # 1) Text
    text_inputs = processor(text=args.text, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(device)

    # 2) Style vector from style_wav (mean-pooled WavLM last hidden state)

    style_wav, sr = sf.read(args.style_wav)
    ssl_inputs = ssl_fe(style_wav, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        for k in ssl_inputs:
            ssl_inputs[k] = ssl_inputs[k].to(device)
        hid = ssl_model(**ssl_inputs).last_hidden_state  # [1, T, 768]
        style_vec = hid.mean(dim=1)  # [1, 768]
        style_latent, _ = style_adaptor(style_vec)       # [1, 256]

    # 3) Speaker x-vector (target voice)
    wav_for_spk, sr2 = sf.read(args.target_spk_wav or args.style_wav)
    with torch.no_grad():
        wav = torch.tensor(wav_for_spk).unsqueeze(0).to(device)
        spk = spk_model.encode_batch(wav)
        spk = torch.nn.functional.normalize(spk, dim=2).squeeze(0)  # [1, 512]
    # 4) Fusion -> 512-D speaker_embeddings for SpeechT5
    fused = fusion(spk, style_latent)  # [1, 512]

    # 5) Generate speech (spectrogram->waveform via vocoder)
    with torch.no_grad():
        speech = tts.generate_speech(input_ids, fused, vocoder=vocoder)  # 16 kHz
    sf.write(args.out_wav, speech.cpu().numpy(), 16000)
    print("Saved:", args.out_wav)


if __name__ == "__main__":
    main()
