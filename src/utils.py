import os
import csv
from typing import List, Dict, Any, Optional

import torch
import soundfile as sf
from datasets import Dataset, Audio

EMOTIONS_CANON = ["neutral","calm","happy","sad","angry","fearful","disgust","surprise"]
EMO_TO_ID = {e: i for i, e in enumerate(EMOTIONS_CANON)}

def read_manifest_csv(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"path", "text", "emotion", "speaker", "style_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        for r in reader:
            r["path"] = os.path.abspath(r["path"])
            r["style_path"] = os.path.abspath(r["style_path"])
            rows.append(r)
    return rows

def build_hf_dataset(rows: List[Dict[str, Any]], sampling_rate: int = 16000) -> Dataset:
    # Create a lightweight Dataset with audio decoding handled by HF Datasets.
    data = {
        "audio": [r["path"] for r in rows],
        "text": [r["text"] for r in rows],
        "emotion": [r["emotion"] for r in rows],
        "emotion_id": [EMO_TO_ID.get(r["emotion"], 0) for r in rows],
        "speaker": [r["speaker"] for r in rows],
        "style_audio": [r["style_path"] for r in rows],
    }
    ds = Dataset.from_dict(data)
    # ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    # ds = ds.cast_column("style_audio", Audio(sampling_rate=sampling_rate))
    return ds

def save_wav(path: str, wav: torch.Tensor, sr: int = 16000):
    wav_np = wav.detach().cpu().numpy()
    sf.write(path, wav_np, sr)
