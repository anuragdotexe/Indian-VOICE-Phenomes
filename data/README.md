Data files for the Indic TTS fine-tuning pipeline.

Expected contents:
- `source_media/` with source MP3/MP4/M4A/WAV files.
- `wav24/` with prepared 24 kHz mono WAV files.
- `raw/` with segmented training clips.
- `transcripts_train.csv` with columns: `path,text,speaker`
- `transcripts_dev.csv` with columns: `path,text,speaker`
- `lexicon.txt` with tab-separated pronunciation overrides.
- `recording_prompts.txt` with balanced sample lines for recording.

Typical workflow:
1. Put source media into `data/source_media/`.
2. Run `python scripts/prepare_source_audio.py` to create `data/wav24/`.
3. Run `python scripts/segment_wav_by_silence.py` to create clips in `data/raw/`.
4. Review and delete bad clips from `data/raw/`.
5. Run `python scripts/make_manifest.py --root data/raw --out data --speaker spk1`.
6. Fill the `text` column in the generated CSVs.
7. Add optional pronunciation overrides to `lexicon.txt`.
