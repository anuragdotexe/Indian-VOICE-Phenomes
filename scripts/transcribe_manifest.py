#!/usr/bin/env python3
"""
Populate the `text` column in a manifest CSV using a local Whisper model.

The first run downloads model weights from Hugging Face unless they are
already cached locally.
"""

import argparse
import csv
import wave
from array import array
from pathlib import Path

import numpy as np
import torch
from transformers import pipeline


def load_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "text", "speaker"])
        writer.writeheader()
        writer.writerows(rows)


def load_wav(path: str):
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"{path} must be 16-bit PCM WAV")

    samples = array("h", frames)
    if channels == 1:
        mono = samples
    else:
        mono = array("h")
        for idx in range(0, len(samples), channels):
            mono.append(sum(samples[idx : idx + channels]) // channels)

    audio = np.array([sample / 32768.0 for sample in mono], dtype=np.float32)
    return {"array": audio, "sampling_rate": sample_rate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--language", default="hi", help='Use "auto" to let the model detect language')
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--start", type=int, default=0, help="Start row offset")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    asr = pipeline("automatic-speech-recognition", model=args.model, device=device, dtype=dtype)

    rows = load_rows(Path(args.input_csv))
    if args.start > 0:
        rows = rows[args.start :]
    if args.limit > 0:
        rows = rows[: args.limit]

    output_rows = []
    for idx, row in enumerate(rows, start=1):
        audio_path = row["path"]
        audio_input = load_wav(audio_path)
        generate_kwargs = {"task": args.task}
        if args.language.lower() != "auto":
            generate_kwargs["language"] = args.language
        result = asr(
            audio_input,
            generate_kwargs=generate_kwargs,
        )
        text = result["text"].strip()
        output_rows.append(
            {
                "path": row["path"],
                "text": text,
                "speaker": row["speaker"],
            }
        )
        print(f"[{idx}/{len(rows)}] {audio_path} -> {text}", flush=True)

    save_rows(Path(args.output_csv), output_rows)
    print(f"Wrote {len(output_rows)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
