import csv, os, re
from pathlib import Path

# Point this to your extracted speech folder:
ROOT = Path("Audio_Speech_Actors_01-24").resolve()

SUFFIX = ""
EMO_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprise"
}
STMT_MAP = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
rows = []
for wav in ROOT.rglob("*.wav"):
    name = wav.stem
    parts = name.split("-")
    if len(parts) != 7:
        continue
    modality, channel, emo, intensity, stmt, rep, actor = parts
    if channel != "01":
        continue
    emotion = EMO_MAP.get(emo)
    text = STMT_MAP.get(stmt)
    if not (emotion and text):
        continue
    wav16 = wav.with_name(wav.stem + SUFFIX)
    path = str(wav16 if wav16.exists() else wav)
    speaker = f"Actor_{actor}"
    rows.append({"path": path, "text": text, "emotion": emotion,
                 "speaker": speaker, "style_path": path})

os.makedirs("data", exist_ok=True)
out_csv = "data/ravdess_manifest.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["path", "text", "emotion", "speaker", "style_path"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Wrote {len(rows)} rows -> {out_csv}")
