"""CLI over the same implementation as the HTTP adapter (serve.respond -> DecisionModel.evaluate).

    python -m decision_model.decide --model runs/showcase/model --request req.json [--device cuda] [--dtype bfloat16]

Prints the /v1/systemone response body as JSON. An invalid or over-limit request prints {"detail": ...} and exits 2.
"""
import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .schema import AdmissionError, DecisionRequest
from .serve import load, respond


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="exported artifact directory")
    ap.add_argument("--request", required=True, help="DecisionRequest JSON file ('-' reads stdin)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    a = ap.parse_args()
    raw = sys.stdin.read() if a.request == "-" else Path(a.request).read_text()
    try:
        req = DecisionRequest.model_validate_json(raw)
        out = respond(load(a.model, a.device, a.dtype), req)
    except (ValidationError, AdmissionError) as e:
        print(json.dumps({"detail": str(e)}), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
