import os, torch, gradio as gr, soundfile as sf
from functools import lru_cache
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, AutoProcessor, WavLMModel
from speechbrain.pretrained import EncoderClassifier
from peft import PeftModel, PeftConfig
from style_modules import StyleAdaptor, StyleSpeakerFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _exists(p): return p and os.path.exists(p)

@lru_cache(maxsize=2)
def load_models(ckpt_dir: str,
                style_adaptor_path: str = None,
                style_fusion_path: str = None):
    ckpt_dir = ckpt_dir or "runs/emotts_esd"
    processor = SpeechT5Processor.from_pretrained(ckpt_dir)
    # Handle PEFT adapter if present in ckpt_dir
    tts = SpeechT5ForTextToSpeech.from_pretrained(ckpt_dir)
    adapter_cfg = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        # If training saved adapters only, wrap base and load adapter on top
        base = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        tts = PeftModel.from_pretrained(base, ckpt_dir)
    tts = tts.to(DEVICE).eval()

    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE).eval()

    style_adaptor = StyleAdaptor()
    fusion = StyleSpeakerFusion()
    sa_path = style_adaptor_path or os.path.join(ckpt_dir, "style_adaptor.pt")
    fu_path = style_fusion_path or os.path.join(ckpt_dir, "style_fusion.pt")
    if not _exists(sa_path) or not _exists(fu_path):
        raise FileNotFoundError("style_adaptor.pt or style_fusion.pt not found in ckpt_dir; did you save them?")
    style_adaptor.load_state_dict(torch.load(sa_path, map_location="cpu"))
    fusion.load_state_dict(torch.load(fu_path, map_location="cpu"))
    style_adaptor, fusion = style_adaptor.to(DEVICE).eval(), fusion.to(DEVICE).eval()

    ssl_proc = AutoProcessor.from_pretrained("microsoft/wavlm-base-plus")
    ssl_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE).eval()

    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
        savedir=os.path.join("/tmp", "spkrec-ecapa-voxceleb")
    )
    return processor, tts, vocoder, style_adaptor, fusion, ssl_proc, ssl_model, spk_model

def synthesize(ckpt_dir, text, style_wav, target_wav=None):
    if not text or not style_wav:
        return None, "Please provide text and a style/reference WAV."
    processor, tts, vocoder, style_adaptor, fusion, ssl_proc, ssl_model, spk_model = load_models(ckpt_dir)

    # 1) Text
    toks = processor(text=text, return_tensors="pt")
    input_ids = toks["input_ids"].to(DEVICE)

    # 2) Style latent
    style_path = style_wav if isinstance(style_wav, str) else style_wav.name
    wav_s, sr = sf.read(style_path)
    ssl_inputs = ssl_proc(audio=wav_s, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        for k in ssl_inputs: ssl_inputs[k] = ssl_inputs[k].to(DEVICE)
        hid = ssl_model(**ssl_inputs).last_hidden_state
        style_latent, _ = style_adaptor(hid.mean(dim=1))

    # 3) Speaker x-vector
    target_path = target_wav if target_wav else style_path
    wav_t, _ = sf.read(target_path)
    with torch.no_grad():
        wav = torch.tensor(wav_t).unsqueeze(0).to(DEVICE)
        spk = torch.nn.functional.normalize(spk_model.encode_batch(wav), dim=2).squeeze(0)

    # 4) Fusion → 512-D
    fused = fusion(spk, style_latent)

    # 5) Generate
    with torch.no_grad():
        speech = tts.generate_speech(input_ids, fused, vocoder=vocoder).cpu().numpy()
    out_path = "gradio_emotts_out.wav"
    sf.write(out_path, speech, 16000)
    return (16000, speech), f"Saved to {out_path}"

def app():
    with gr.Blocks(title="Emotion-aware TTS Style Transfer") as demo:
        gr.Markdown("## Emotion-aware TTS (SpeechT5 + Style Transfer)")
        with gr.Row():
            ckpt_dir = gr.Textbox(label="Checkpoint dir", value="runs/emotts_esd")
        text = gr.Textbox(label="Text to synthesize", value="I can't believe this happened.")
        style = gr.Audio(label="Style / reference WAV (emotion/prosody)", type="filepath")
        target = gr.Audio(label="Optional target speaker WAV (voice)", type="filepath")
        btn = gr.Button("Synthesize")
        out_audio = gr.Audio(label="Output", type="numpy")
        status = gr.Markdown()

        btn.click(fn=synthesize,
                  inputs=[ckpt_dir, text, style, target],
                  outputs=[out_audio, status])
    return demo

if __name__ == "__main__":
    app().queue(api_open=False).launch()
