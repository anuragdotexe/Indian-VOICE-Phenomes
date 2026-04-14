import argparse, yaml, torch, asyncio, json
import torchaudio
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC

app = FastAPI()

def load(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["output_dir"])
    model = AutoModelForCausalLM.from_pretrained(cfg["output_dir"], device_map="auto")
    codec = SNAC.from_pretrained(cfg["base_model"])
    return tok, model, codec

@app.on_event("startup")
async def startup():
    global TOK, MODEL, CODEC, CFG
    CFG = app.state.cfg
    TOK, MODEL, CODEC = load(CFG)

@app.post("/synthesize")
async def synthesize(payload: dict):
    text = payload["text"]
    ids = TOK(text, return_tensors="pt").to(MODEL.device)
    codes = MODEL.generate(**ids, max_new_tokens=800)
    wav = CODEC.decode(codes[0]).cpu()
    def iter_audio():
        buf = torchaudio.save(torch.ops.torchaudio.io.StreamWriter(), wav, CFG["sample_rate"], format="wav")  # placeholder
        yield buf
    return StreamingResponse(iter_audio(), media_type="audio/wav")

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        text = json.loads(data)["text"]
        ids = TOK(text, return_tensors="pt").to(MODEL.device)
        codes = MODEL.generate(**ids, max_new_tokens=800, do_sample=True)
        wav = CODEC.decode(codes[0]).cpu().numpy()
        await ws.send_bytes(wav.tobytes())

def run(cfg):
    import uvicorn
    app.state.cfg = cfg
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
