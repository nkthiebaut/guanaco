from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


model = ChatOpenAI(model="gpt-3.5-turbo")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant who speaks like a pirate"),
        ("human", "{text}"),
    ]
)


chain = prompt | model


add_routes(app, chain, path="/pirate")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
