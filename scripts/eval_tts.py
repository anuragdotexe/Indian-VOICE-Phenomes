import argparse, yaml, os, torch, torchaudio
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC  # lightweight codec/decoder
import jiwer

def synth(model, tok, codec, text, device):
    ids = tok(text, return_tensors="pt").to(device)
    codes = model.generate(**ids, max_new_tokens=800)
    wav = codec.decode(codes[0]).cpu()
    return wav

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["output_dir"])
    model = AutoModelForCausalLM.from_pretrained(cfg["output_dir"], device_map="auto")
    codec = SNAC.from_pretrained(cfg["base_model"])

    ds = load_dataset("csv", data_files=cfg["dev_csv"])["train"]
    refs, hyps = [], []
    os.makedirs("outputs/dev_audio", exist_ok=True)

    for i, row in enumerate(ds):
        wav = synth(model, tok, codec, row["text"], device)
        path = f"outputs/dev_audio/{i:03d}.wav"
        torchaudio.save(path, wav, cfg["sample_rate"])
        # crude ASR eval via Whisper tiny.en (optional)
        # Add your ASR call here; placeholder below
        hypo = row["text"]  # replace with ASR transcription
        refs.append(row["text"]); hyps.append(hypo)

    print("WER placeholder:", jiwer.wer(refs, hyps))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
