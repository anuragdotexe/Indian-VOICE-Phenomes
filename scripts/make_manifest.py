#!/usr/bin/env python3
"""
Scan a directory tree of audio files and build TTS manifest CSVs.

Outputs in the --out folder (default: data/):
- transcripts_all.csv  (path,text,speaker)
- transcripts_train.csv
- transcripts_dev.csv

Usage examples:
  python scripts/make_manifest.py --root /abs/path/to/audio --out data --dev-frac 0.1 --ext wav flac mp3
  python scripts/make_manifest.py --root data/raw --speaker spk1

Notes:
- Paths in CSV are written relative to --root so you can move the folder.
- speaker defaults to parent folder name; override with --speaker for single speaker.
- text is left blank; fill transcripts manually after generation.
"""
import argparse
import csv
import random
from pathlib import Path


def iter_audio(root: Path, exts):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower().lstrip(".") in exts:
            yield path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing audio files")
    ap.add_argument("--out", default="data", help="Output folder for CSVs")
    ap.add_argument("--ext", nargs="+", default=["wav", "flac", "mp3"], help="Audio extensions to include")
    ap.add_argument("--speaker", default=None, help="Override speaker id for all rows")
    ap.add_argument("--dev-frac", type=float, default=0.1, help="Fraction for dev split (0 to disable)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling before split")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    exts = {e.lower().lstrip(".") for e in args.ext}
    rows = []
    for p in iter_audio(root, exts):
        rel = p.relative_to(root)
        speaker = args.speaker or p.parent.name
        rows.append({"path": str(rel), "text": "", "speaker": speaker})

    if not rows:
        raise SystemExit(f"No audio found under {root} with extensions {sorted(exts)}")

    random.Random(args.seed).shuffle(rows)

    # write all
    all_csv = out / "transcripts_all.csv"
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "text", "speaker"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {all_csv}")

    # split train/dev
    dev_n = max(1, int(len(rows) * args.dev_frac)) if args.dev_frac > 0 else 0
    dev_rows = rows[:dev_n]
    train_rows = rows[dev_n:]

    train_csv = out / "transcripts_train.csv"
    with train_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "text", "speaker"])
        w.writeheader(); w.writerows(train_rows)

    if dev_rows:
        dev_csv = out / "transcripts_dev.csv"
        with dev_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "text", "speaker"])
            w.writeheader(); w.writerows(dev_rows)
        print(f"Train: {len(train_rows)}  Dev: {len(dev_rows)} -> {dev_csv}")
    else:
        print(f"Train: {len(train_rows)}  Dev: 0 (dev split disabled)")

    print("Next: open the CSVs and fill the 'text' column with transcripts. Keep path/speaker unchanged.")


if __name__ == "__main__":
    main()
