import argparse, os, csv, re, random
from pathlib import Path
from collections import defaultdict

AUDIO_EXTS = {".wav", ".flac", ".mp3"}

EMO_CANON = {"neutral","happy","angry","sad","surprise","amused","disgust","sleepy"}

def load_text_map(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"text_map not found: {path}")
    m = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not ({"path","text"} <= cols or {"utt_id","text"} <= cols):
            raise ValueError("text_map must have either {path,text} or {utt_id,text} columns")
        by_path = "path" in cols
        for r in reader:
            key = r["path"] if by_path else r["utt_id"]
            m[key] = r["text"]
    return m

def parse_txt_done_data(root: Path):
    """
    Collect all CMU Arctic style 'txt.done.data' mappings under root.
    Each line looks like: (arctic_a0001) THIS IS A SENTENCE.
    Returns dict: {utt_id: text}
    """
    mapping = {}
    for p in root.rglob("txt.done.data"):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                m = re.match(r"\(([^)]+)\)\s*(.+)", line)
                if m:
                    utt_id, text = m.group(1), m.group(2).strip()
                    mapping[utt_id] = text
    return mapping

def guess_emotion_from_path(parts):
    # Look for a directory segment that matches a known emotion (case-insensitive)
    for seg in parts[::-1]:
        s = seg.lower()
        for e in EMO_CANON:
            if s == e or s.startswith(e):
                return e
    return None

def find_speaker_from_path_esd(path: Path):
    # ESD: .../<English|Chinese>/<speaker>/<emotion>/<file>
    parts = path.parts
    for i, seg in enumerate(parts):
        if seg.lower() in {"english","chinese"} and i+2 < len(parts):
            return parts[i+1]
    return parts[-3] if len(parts) >= 3 else "spk"

def find_speaker_from_path_emovdb(path: Path):
    # EmoV-DB: .../<speaker>/<emotion>/<file>
    return path.parts[-3] if len(path.parts) >= 3 else "spk"

def collect_audio_files(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]

def build_rows_esd(root: Path):
    rows = []
    for wav in collect_audio_files(root):
        emo = guess_emotion_from_path(list(wav.parent.parts))
        spk = find_speaker_from_path_esd(wav)
        rows.append({"path": str(wav), "emotion": emo or "neutral", "speaker": spk})
    return rows

def build_rows_emovdb(root: Path):
    rows = []
    for wav in collect_audio_files(root):
        emo = guess_emotion_from_path(list(wav.parent.parts))
        spk = find_speaker_from_path_emovdb(wav)
        rows.append({"path": str(wav), "emotion": emo or "neutral", "speaker": spk})
    return rows

def build_rows_generic(root: Path):
    """
    Generic: assume .../<speaker>/<emotion>/<*.wav>
    """
    rows = []
    for wav in collect_audio_files(root):
        emo = guess_emotion_from_path(list(wav.parent.parts)) or "neutral"
        spk = wav.parts[-3] if len(wav.parts) >= 3 else "spk"
        rows.append({"path": str(wav), "emotion": emo, "speaker": spk})
    return rows

def extract_keys_for_text(path: Path):
    """
    Returns candidate keys to resolve text from maps:
    - full path as string
    - basename without extension
    - arctic_* id (e.g., arctic_a0001) if present
    - a0001/b0001 like ids if present
    """
    stem = path.stem
    cands = [str(path), stem]
    m = re.search(r"(arctic_[ab]\d{4})", stem, flags=re.IGNORECASE)
    if m: cands.append(m.group(1))
    m2 = re.search(r"([ab]\d{4})", stem, flags=re.IGNORECASE)
    if m2: cands.append(m2.group(1))
    return cands

def pick_style(row, pool_by_key, style_mode, rng):
    # pool_by_key: dict key -> list of indices with same property
    if style_mode == "same":
        return row["path"]
    key = None
    if style_mode == "random_same_emotion":
        key = ("emo", row["emotion"])
    elif style_mode == "random_within_speaker":
        key = ("spk", row["speaker"])
    elif style_mode == "random_global":
        key = ("global", "all")
    idxs = pool_by_key.get(key, [])
    if not idxs:
        return row["path"]
    choice = rng.choice(idxs)
    return choice["path"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder of dataset")
    ap.add_argument("--dataset", choices=["esd","emovdb","generic"], required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--style_mode", choices=["same","random_same_emotion","random_within_speaker","random_global"], default="same")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--text_map", default=None, help="CSV with {path,text} or {utt_id,text}")
    ap.add_argument("--auto_cmu_arctic", action="store_true", help="Parse any txt.done.data files under --root")
    ap.add_argument("--allow_missing_text", action="store_true", help="If set, missing text rows will be skipped instead of erroring")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    rng = random.Random(args.seed)

    if args.dataset == "esd":
        base_rows = build_rows_esd(root)
    elif args.dataset == "emovdb":
        base_rows = build_rows_emovdb(root)
    else:
        base_rows = build_rows_generic(root)

    # Build pools for style sampling
    pool_by_key = defaultdict(list)
    for r in base_rows:
        pool_by_key[("emo", r["emotion"])].append(r)
        pool_by_key[("spk", r["speaker"])].append(r)
        pool_by_key[("global","all")].append(r)

    # Load text maps
    text_map = {}
    if args.text_map:
        text_map.update(load_text_map(args.text_map))
    if args.auto_cmu_arctic:
        text_map.update(parse_txt_done_data(root))

    # Resolve text + style_path
    out_rows = []
    missing = 0
    for r in base_rows:
        wav = Path(r["path"])
        # Resolve text
        text = None
        for k in extract_keys_for_text(wav):
            if k in text_map:
                text = text_map[k]
                break
        if text is None:
            if args.allow_missing_text:
                missing += 1
                continue
            else:
                raise ValueError(f"No text found for {wav}. Provide --text_map or use --allow_missing_text")
        # style
        style_path = pick_style(r, pool_by_key, args.style_mode, rng)
        out_rows.append({
            "path": str(wav),
            "text": text,
            "emotion": r["emotion"],
            "speaker": r["speaker"],
            "style_path": style_path
        })

    os.makedirs(Path(args.out_csv).parent, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path","text","emotion","speaker","style_path"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote {len(out_rows)} rows → {args.out_csv}" + (f" (skipped {missing} without text)" if missing else ""))

if __name__ == "__main__":
    main()
