#!/usr/bin/env python3
"""
Convert source MP3/MP4/M4A audio into 24 kHz mono WAV files.

Uses macOS `afconvert`, which is available by default.
"""

import argparse
import shutil
import subprocess
from pathlib import Path


SUPPORTED_EXTS = {".mp3", ".mp4", ".m4a", ".wav"}


def convert_with_afconvert(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "afconvert",
        "-f",
        "WAVE",
        "-d",
        "LEI16@24000",
        "-c",
        "1",
        str(src),
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def maybe_copy_wav(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="data/source_media",
        help="Folder containing source MP3/MP4/M4A/WAV files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/wav24",
        help="Folder to write 24 kHz mono WAV files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    )
    if not sources:
        raise SystemExit(f"No supported media files found in {input_dir}")

    for src in sources:
        dst = output_dir / f"{src.stem}.wav"
        if src.suffix.lower() == ".wav":
            maybe_copy_wav(src, dst)
        else:
            convert_with_afconvert(src, dst)
        print(f"Prepared {dst}")


if __name__ == "__main__":
    main()
