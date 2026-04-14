import argparse, os, yaml, random, torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
import torchaudio

def load_audio(path, sr):
    wav, s = torchaudio.load(path)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    return wav.mean(0, keepdim=True)

def collate(batch, tok, sr):
    texts = [b["text"] for b in batch]
    wavs = [load_audio(b["path"], sr) for b in batch]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
    max_len = max(w.shape[-1] for w in wavs)
    audio = torch.zeros(len(wavs), 1, max_len)
    for i, w in enumerate(wavs):
        audio[i, 0, : w.shape[-1]] = w
    return {**enc, "audio": audio}

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=cfg["lora_dropout"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("csv", data_files=cfg["train_csv"])["train"]
    dl = DataLoader(ds, batch_size=cfg["batch_size"],
                    shuffle=True, collate_fn=lambda b: collate(b, tok, cfg["sample_rate"]))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    model.train()
    for epoch in range(cfg["epochs"]):
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, labels=batch["input_ids"])
            out.loss.backward()
            if (step + 1) % cfg["grad_accum"] == 0:
                opt.step(); opt.zero_grad()
            if step % cfg["log_every_steps"] == 0:
                print(f"e{epoch} s{step} loss {out.loss.item():.3f}")
        model.save_pretrained(os.path.join(cfg["output_dir"], f"epoch{epoch}"))
    tok.save_pretrained(cfg["output_dir"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    main(cfg)
