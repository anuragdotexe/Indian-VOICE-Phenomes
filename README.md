# Indian-VOICE-Phenomes

Self-hosted Indic voice adaptation workspace for building an Indian-accent TTS pipeline without SaaS APIs.

## Current status

- Dataset preparation pipeline is implemented locally.
- Source audio was converted, segmented, and reduced into a seed dataset.
- Draft seed transcripts were generated with local Whisper STT.
- Large model/audio assets are kept local and ignored by git.
- Svara model weights are downloaded locally under `models/`.
- Fine-tuning is **not started yet** because the current training script is not aligned with the downloaded Svara checkpoint format.

## Repository layout

```text
.
├── configs/
├── data/
├── models/                  # local only, gitignored
├── runs/                    # local only, training outputs
├── scripts/
└── svara-tts-v1/            # upstream nested repo / reference checkout
```

## What is in version control

Tracked:

- lightweight project scripts
- dataset manifests / draft transcript CSVs
- small config and documentation files

Not tracked:

- `models/`
- `data/raw/`
- `data/source_media/`
- `data/wav24/`
- `data/text/`
- `.DS_Store`

These are ignored in [.gitignore](/Users/anuragroy/LLM-Voice/.gitignore:1) to avoid GitHub LFS size-limit failures.

## Dataset pipeline implemented so far

### 1. Source collection

Two clean mono source recordings were used as the seed accent/style reference:

- `If_You_re_Single___Sad__Watch_This___BeerBiceps_clean.mp3`
- `Yeh_Kahaani_Aapke_Ander_Aag_Laga_Degi___Mahabharata_Focus_Motivation__The_Ranveer_Show____26_clean.mp3`

### 2. Audio preparation

Implemented in [prepare_source_audio.py](/Users/anuragroy/LLM-Voice/scripts/prepare_source_audio.py:1):

- converts source media to `24 kHz` mono WAV
- writes normalized audio into `data/wav24/`

### 3. Clip segmentation

Implemented in [segment_wav_by_silence.py](/Users/anuragroy/LLM-Voice/scripts/segment_wav_by_silence.py:1):

- splits long WAV files into short candidate utterances
- writes segmented clips into `data/raw/`

### 4. Manifest generation

Implemented in [make_manifest.py](/Users/anuragroy/LLM-Voice/scripts/make_manifest.py:1):

- walks an audio tree
- creates CSV manifests with columns:
  - `path`
  - `text`
  - `speaker`

### 5. Subset generation

Implemented in [create_subset_manifest.py](/Users/anuragroy/LLM-Voice/scripts/create_subset_manifest.py:1):

- reduces a large dataset into smaller train/dev subsets
- was used to create a manageable seed set for first-pass training

### 6. STT-assisted transcript drafting

Implemented in [transcribe_manifest.py](/Users/anuragroy/LLM-Voice/scripts/transcribe_manifest.py:1):

- runs local Whisper-based STT over a manifest CSV
- writes draft transcript CSVs for correction

## Active dataset files

The current seed files to review/correct are:

- [data/transcripts_train_seed_autotext.csv](/Users/anuragroy/LLM-Voice/data/transcripts_train_seed_autotext.csv:1)
- [data/transcripts_dev_seed_autotext.csv](/Users/anuragroy/LLM-Voice/data/transcripts_dev_seed_autotext.csv:1)

Supporting files:

- [data/README.md](/Users/anuragroy/LLM-Voice/data/README.md:1)
- [data/lexicon.txt](/Users/anuragroy/LLM-Voice/data/lexicon.txt:1)
- [data/recording_prompts.txt](/Users/anuragroy/LLM-Voice/data/recording_prompts.txt:1)

## Model status

The Svara weights are downloaded locally in:

- `models/svara-tts-v1/`

However, the current checkpoint loads as a plain `LlamaForCausalLM`, and the current [train_lora.py](/Users/anuragroy/LLM-Voice/scripts/train_lora.py:1) script expects a raw-waveform training path that does **not** match the discrete-audio-token architecture described by Svara.

Because of that:

- the repo is ready for dataset work
- the repo is **not yet ready for correct Svara fine-tuning**

## Known blocker

The main blocker is the training stack:

- Svara inference assets are available
- an official public fine-tuning pipeline for this checkpoint was not found
- the current local `train_lora.py` is not a validated Svara training recipe

This means the next engineering decision is one of:

1. build or adapt a proper SNAC-token fine-tuning pipeline for Svara
2. switch to a different open TTS model with public fine-tuning code
3. use Svara for inference only and postpone custom fine-tuning

## Scripts currently present

- [scripts/train_lora.py](/Users/anuragroy/LLM-Voice/scripts/train_lora.py:1)
- [scripts/eval_tts.py](/Users/anuragroy/LLM-Voice/scripts/eval_tts.py:1)
- [scripts/serve_ws.py](/Users/anuragroy/LLM-Voice/scripts/serve_ws.py:1)
- [scripts/make_manifest.py](/Users/anuragroy/LLM-Voice/scripts/make_manifest.py:1)
- [scripts/prepare_source_audio.py](/Users/anuragroy/LLM-Voice/scripts/prepare_source_audio.py:1)
- [scripts/segment_wav_by_silence.py](/Users/anuragroy/LLM-Voice/scripts/segment_wav_by_silence.py:1)
- [scripts/create_subset_manifest.py](/Users/anuragroy/LLM-Voice/scripts/create_subset_manifest.py:1)
- [scripts/transcribe_manifest.py](/Users/anuragroy/LLM-Voice/scripts/transcribe_manifest.py:1)

## Recommended next step

Before any fine-tuning run:

1. finalize transcript corrections in the seed CSVs
2. decide whether to stay on Svara or switch to a model with public training support
3. validate the actual training path on a tiny dry run

Until that is done, this repo should be treated as:

- a prepared local dataset and preprocessing workspace
- not yet a complete end-to-end fine-tuning pipeline
