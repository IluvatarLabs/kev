# Modified implementation for Kev; see LICENSE and NOTICE.
"""HTTP adapter over DecisionModel.evaluate: one loaded artifact, Jev-shaped routes, nothing else.

    python -m decision_model.serve --model runs/showcase/model [--port 8008] [--device cuda] [--dtype bfloat16]

POST /v1/systemone   body = DecisionRequest -> {"model", "answers", "usage", "latency_ms"}; an admission failure (state,
                     row, question or option limits) or an invalid body -> 422 with the message and no answers.
GET  /v1/models      the artifact path, backbone, dtype, temperature and admission limits.
Every response carries `x-typesafe-request-id` (echoed or generated). Bearer auth is required when DECISION_MODEL_API_KEY
is set; unset = open server (the local default). The CLI (decision_model.decide) calls respond() below, so the Python,
CLI and HTTP paths share one implementation.
"""
import argparse
import hmac
import json
import os
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .schema import AdmissionError, DecisionRequest

API_KEY_ENV = "DECISION_MODEL_API_KEY"


def load(path, device=None, dtype=None):
    import torch
    from .model import DecisionModel
    return DecisionModel.from_pretrained(path, device=device, dtype=getattr(torch, dtype) if dtype else None)


def respond(model, req: DecisionRequest) -> dict:
    """The /v1/systemone body for one validated request. Raises AdmissionError for an over-limit request."""
    out = model.evaluate(state=req.state, questions={k: q.model_dump() for k, q in req.questions.items()})
    return {"model": req.model, "answers": out["answers"], "usage": out["usage"], "latency_ms": out["latency_ms"]}


def create_app(model, path) -> FastAPI:
    app = FastAPI(title="decision-model")
    lock = threading.Lock()   # one model, one request on the device at a time
    cfg = json.loads((Path(path) / "config.json").read_text())
    api_key = os.environ.get(API_KEY_ENV)

    @app.middleware("http")
    async def typesafe(request, call_next):
        if api_key and request.url.path.startswith("/v1") and not hmac.compare_digest(
                request.headers.get("authorization", ""), f"Bearer {api_key}"):
            resp = JSONResponse({"detail": f"missing or invalid API key; send Authorization: Bearer <{API_KEY_ENV}>"}, 401,
                                {"www-authenticate": "Bearer"})
        else:
            resp = await call_next(request)
        resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
        return resp

    @app.post("/v1/systemone")
    def systemone(req: DecisionRequest):
        try:
            with lock:
                return respond(model, req)
        except AdmissionError as e:
            return JSONResponse({"detail": str(e)}, 422)

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": "decision-model", "path": str(path), "backbone": cfg["backbone"], "parent": cfg["parent"],
                            "device": str(model.device), "dtype": str(next(model.lm.parameters()).dtype).removeprefix("torch."),
                            "temperature": model.head.temperature, "limits": cfg["limits"]}]}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="exported artifact directory")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: cuda when available, else cpu)")
    ap.add_argument("--dtype", default=None, help="bfloat16 | float32 (default: the artifact's dtype)")
    a = ap.parse_args()
    model = load(a.model, a.device, a.dtype)
    import uvicorn
    print(f"serving {a.model} on {model.device} at http://127.0.0.1:{a.port}/v1/systemone", flush=True)
    uvicorn.run(create_app(model, a.model), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
