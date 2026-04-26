#!/usr/bin/env python3
"""
Split 16-bit mono WAV files into shorter utterances using simple silence detection.
"""

import argparse
import math
import wave
from array import array
from pathlib import Path


def read_wav(path: Path):
    with wave.open(str(path), "rb") as wav:
        params = wav.getparams()
        if params.nchannels != 1 or params.sampwidth != 2:
            raise ValueError(f"{path} must be 16-bit mono WAV")
        frames = wav.readframes(params.nframes)
    samples = array("h", frames)
    return params.framerate, samples


def rms(chunk: array) -> float:
    if not chunk:
        return 0.0
    power = sum(sample * sample for sample in chunk) / len(chunk)
    return math.sqrt(power)


def segment_samples(samples: array, sr: int, silence_ms: int, min_ms: int, max_ms: int, threshold: float):
    frame_ms = 20
    frame_size = max(1, int(sr * frame_ms / 1000))
    silence_frames = max(1, silence_ms // frame_ms)
    min_frames = max(1, min_ms // frame_ms)
    max_frames = max(1, max_ms // frame_ms)

    segments = []
    current_start = None
    silent_run = 0

    total_frames = math.ceil(len(samples) / frame_size)
    for frame_idx in range(total_frames):
        start = frame_idx * frame_size
        end = min(len(samples), start + frame_size)
        chunk = samples[start:end]
        voiced = rms(chunk) >= threshold

        if voiced and current_start is None:
            current_start = frame_idx
            silent_run = 0
            continue

        if current_start is None:
            continue

        if voiced:
            silent_run = 0
        else:
            silent_run += 1

        current_len = frame_idx - current_start + 1
        if current_len >= max_frames:
            segments.append((current_start, frame_idx + 1))
            current_start = None
            silent_run = 0
            continue

        if silent_run >= silence_frames and current_len >= min_frames:
            segments.append((current_start, frame_idx - silence_frames + 1))
            current_start = None
            silent_run = 0

    if current_start is not None:
        end_frame = total_frames
        if end_frame - current_start >= min_frames:
            segments.append((current_start, end_frame))

    return [
        (start * frame_size, min(len(samples), end * frame_size))
        for start, end in segments
        if end > start
    ]


def write_segment(path: Path, sr: int, chunk: array):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(chunk.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/wav24", help="Folder with prepared WAV files")
    parser.add_argument("--output-dir", default="data/raw", help="Folder to write segmented clips")
    parser.add_argument("--silence-ms", type=int, default=450)
    parser.add_argument("--min-ms", type=int, default=1800)
    parser.add_argument("--max-ms", type=int, default=8000)
    parser.add_argument("--threshold", type=float, default=650.0, help="RMS threshold for speech detection")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No WAV files found in {input_dir}")

    for wav_path in wav_files:
        sr, samples = read_wav(wav_path)
        segments = segment_samples(
            samples,
            sr,
            silence_ms=args.silence_ms,
            min_ms=args.min_ms,
            max_ms=args.max_ms,
            threshold=args.threshold,
        )
        print(f"{wav_path.name}: {len(segments)} segments")
        for idx, (start, end) in enumerate(segments, start=1):
            clip = array("h", samples[start:end])
            out_path = Path(args.output_dir) / wav_path.stem / f"{wav_path.stem}_{idx:03d}.wav"
            write_segment(out_path, sr, clip)


if __name__ == "__main__":
    main()
