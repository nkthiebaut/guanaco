import sys
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from einops import rearrange, reduce
import torch
from guanaco.model import Guanaco

logger.remove()
logger.add(sys.stderr, level="INFO", format="{message}", serialize=False)
logger.info("Starting API, model version v0...")


model = Guanaco(vocab_size=256, emb_dim=512, n_heads=1, n_layers=1, max_len=128)
state_dict = torch.load("./checkpoint.ckpt")["state_dict"]
state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

logger.info("Model loaded...")

app = FastAPI()


class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50


class GeneratedText(BaseModel):
    generated_text: str


@app.get("/")
def get_root() -> dict:
    logger.info("Received request on the root endpoint")
    return {"status": "ok"}


def tokenize(text: str) -> list[int]:
    """Converts a string to a bytes object using UTF-8 encoding."""
    return [code_unit for code_unit in text.encode("utf-8")]


def detokenize(token_ids: list[int]) -> str:
    """Converts a bytes object to a string using UTF-8 encoding."""
    return bytes(token_ids).decode("utf-8", errors="replace")


def generate(model, x: str, n_tokens: int = 5, device="cpu"):
    """Predict next token with greedy decoding."""
    x = torch.tensor(tokenize(x)).unsqueeze(0)
    x = x.to(device)
    model = model.to(device)

    for _ in range(n_tokens):
        pred = model(x)[:, -1, :]  # Logits of the next token prediction (B, V)
        next_tokens = pred.argmax(dim=-1)  # Next token_id with highest proba (B)
        next_tokens = rearrange(next_tokens, "B -> B 1")
        x = torch.cat((x, next_tokens), dim=1)
    return "".join(detokenize(x[0].tolist()))


@app.post("/generate", response_model=GeneratedText)
async def generate_text(request: TextGenerationRequest):
    try:
        generated_text = generate(model, request.prompt, request.max_length)
        logger.info(f"Generated text: {generated_text}")
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
