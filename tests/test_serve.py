"""HTTP and CLI serve the same implementation on a real exported artifact (DM_TEST_MODEL, default runs/plumbing/model)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MODEL = Path(os.environ.get("DM_TEST_MODEL", "runs/plumbing/model"))
pytestmark = pytest.mark.skipif(not (MODEL / "config.json").exists(), reason=f"no exported artifact at {MODEL} (set DM_TEST_MODEL)")

# questions and options the model never saw in training
REQUEST = {"state": {"ticket": "My invoice shows a charge for a premium add-on I never ordered. Please remove it before Friday."},
           "questions": {"department": {"type": "choice", "instructions": "Route this ticket to one team.",
                                        "criteria": {"billing": "Invoices and charges", "tech": "Outages and bugs",
                                                     "sales": "New purchases", "legal": None}},
                         "deadline": {"type": "noul", "instructions": "Does the customer state a deadline?"},
                         "severity": {"type": "score", "instructions": "How severe is the problem?",
                                      "criteria": ["cosmetic", "annoying", "blocking", "critical", "outage"]}}}


def test_http_and_cli_answer_new_schemas_and_refuse_overflow(tmp_path):
    from fastapi.testclient import TestClient
    from decision_model.serve import create_app, load

    client = TestClient(create_app(load(MODEL), MODEL))
    r = client.post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["answers"]) == set(REQUEST["questions"])
    expected = {"department": ["billing", "tech", "sales", "legal"], "deadline": ["false", "true"], "severity": ["0", "1", "2", "3", "4"]}
    for qid, keys in expected.items():
        probs = body["answers"][qid]["probabilities"]
        assert list(probs) == keys
        assert abs(sum(probs.values()) - 1) < 1e-6

    # a state over 2,048 tokens is refused, never truncated
    over = {**REQUEST, "state": "word " * 2100}
    r = client.post("/v1/systemone", json=over)
    assert r.status_code == 422 and "answers" not in r.json() and "limit" in r.json()["detail"]

    # the CLI prints the same answers for the same request
    req = tmp_path / "req.json"
    req.write_text(json.dumps(REQUEST))
    out = subprocess.run([sys.executable, "-m", "decision_model.decide", "--model", str(MODEL), "--request", str(req)],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout)["answers"] == body["answers"]
