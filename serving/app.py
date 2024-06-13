import sys
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

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


@app.post("/generate", response_model=GeneratedText)
async def generate_text(request: TextGenerationRequest):
    try:
        # TODO: implement the generation logic
        generated_text = "PLACEHOLDER TEXT"
        logger.info(f"Generated text: {generated_text}")
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
