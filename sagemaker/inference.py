import io, os, json, base64
from typing import Any, Dict
import numpy as np
import torch

try:
    import soundfile as sf
except Exception:
    sf = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model(model_dir: str):
    # Try torchscript first
    ts = [os.path.join(model_dir, n) for n in ("model.ts", "model.torchscript.pt")]
    for p in ts:
        if os.path.exists(p):
            return torch.jit.load(p, map_location=DEVICE).eval()
    pth = os.path.join(model_dir, "model.pth")
    if os.path.exists(pth):
        state = torch.load(pth, map_location=DEVICE)
        state = state.get("state_dict", state)
        model = torch.nn.Linear(8, 8)  # minimal stub if your class isn't importable
        model.load_state_dict(state, strict=False)
        return model.eval()
    # Last resort: presence-only artifact
    pt = os.path.join(model_dir, "model.pt")
    if os.path.exists(pt):
        return torch.load(pt, map_location=DEVICE)
    raise RuntimeError("No model artifact found in model_dir")

def model_fn(model_dir: str):
    return _load_model(model_dir)

def input_fn(body: bytes, content_type: str):
    if content_type and "application/json" in content_type:
        return json.loads(body)
    if content_type and "text/plain" in content_type:
        return {"text": body.decode("utf-8"), "emotion": "neutral"}
    raise ValueError(f"Unsupported content type: {content_type}")

def _synthesize(model, text: str, emotion: str, sample_rate: int = 22050):
    """
    Replace with your real forward pass. We keep a sine-wave fallback
    so the endpoint is demoable before wiring the TTS pipeline.
    """
    try:
        # If you expose a project synth like src.tts.pipeline.synthesize(...)
        import importlib
        for modname in ("src.tts.pipeline", "src.infer", "src.synthesis"):
            try:
                mod = importlib.import_module(modname)
                if hasattr(mod, "synthesize"):
                    wav = mod.synthesize(model, text=text, emotion=emotion, sample_rate=sample_rate)
                    return np.asarray(wav, dtype=np.float32), sample_rate
            except Exception:
                continue
    except Exception:
        pass
    # Fallback sine
    duration = 0.7
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq = {"angry": 600, "happy": 880, "sad": 220, "neutral": 440}.get(emotion, 440)
    return (0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32), sample_rate)

def predict_fn(inputs: Dict[str, Any], model):
    text = inputs.get("text", "")
    emotion = inputs.get("emotion", "neutral")
    sr = int(inputs.get("sample_rate", 22050))
    wav, sr = _synthesize(model, text, emotion, sr)
    return {"wav": wav, "sr": sr}

def output_fn(pred, accept: str):
    wav, sr = pred["wav"], pred["sr"]
    if accept and "audio/wav" in accept:
        if sf is None: raise RuntimeError("Install 'soundfile' to return WAV bytes.")
        buf = io.BytesIO(); sf.write(buf, wav, sr, format="WAV")
        return buf.getvalue(), "audio/wav"
    # default JSON (base64 wav or raw PCM if soundfile absent)
    if sf is None:
        return json.dumps({"sample_rate": sr, "pcm_f32_base64": base64.b64encode(wav.tobytes()).decode()}), "application/json"
    buf = io.BytesIO(); sf.write(buf, wav, sr, format="WAV")
    return json.dumps({"sample_rate": sr, "wav_base64": base64.b64encode(buf.getvalue()).decode()}), "application/json"
