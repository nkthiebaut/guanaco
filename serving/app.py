import sys
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


logger.remove()
logger.add(sys.stderr, level="INFO", format="{message}", serialize=False)
logger.info("Starting API, model version v0...")


# Load the BLOOM model and tokenizer
model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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
        inputs = tokenizer(request.prompt, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=request.max_length)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
