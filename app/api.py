"""Optional REST API:  uvicorn app.api:app --reload   (docs at /docs)"""
from fastapi import FastAPI
from pydantic import BaseModel
from .service import Copilot

app = FastAPI(title="Enterprise Intelligence Copilot")
copilot = Copilot()


class Ask(BaseModel):
    question: str
    role: str = "viewer"
    mode: str = "hybrid"


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(copilot.chunks)}


@app.post("/ask")
def ask(body: Ask):
    return copilot.ask(body.question, body.role, body.mode)
