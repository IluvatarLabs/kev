"""Export equivalence on a real training run (DM_TEST_RUN, default runs/plumbing/run): pre-merge, merged and the offline
reloaded artifact agree; the reloaded model answers new questions/options with the parent unreachable; the artifact holds
the complete 28-layer backbone and no vocabulary projection."""
import json
import os
import re
from pathlib import Path

import pytest
import torch

RUN = Path(os.environ.get("DM_TEST_RUN", "runs/plumbing/run"))
pytestmark = pytest.mark.skipif(not (RUN / "run.json").exists() or not torch.cuda.is_available(),
                                reason=f"needs a training run at {RUN} (set DM_TEST_RUN) and CUDA")

NEW = {"state": "Order #5521: the blender stopped working after two days and the customer wants it collected.",
       "questions": {"action": {"type": "choice", "instructions": "Which action should the agent take first?",
                                "criteria": {"pickup": "Arrange a courier collection", "manual": "Send the troubleshooting guide",
                                             "credit": "Issue store credit", "close": "Close the ticket", "escalate": None}},
                     "warranty": {"type": "noul", "instructions": "Is the product likely under warranty?"},
                     "effort": {"type": "score", "instructions": "How much agent effort will this take?",
                                "criteria": ["minimal", "low", "moderate", "high", "extreme", "unknown-scale"]}}}


def test_export_reproduces_run_offline(tmp_path):
    from safetensors import safe_open
    from decision_model.export import export

    out = tmp_path / "model"
    report = export(RUN, out, calibration={"temperature": 1.3, "fitting_scope": "test", "counts": {}}, conversion={},
                    device="cuda", check_requests=[NEW])
    # export() raises on the tolerances; reloaded in a fresh process with the network and the parent unavailable, the
    # artifact answers every new option
    answers = report["reloaded_answers"][0]
    assert list(answers["action"]["probabilities"]) == list(NEW["questions"]["action"]["criteria"])
    assert list(answers["effort"]["probabilities"]) == [str(i) for i in range(6)]
    assert abs(sum(answers["warranty"]["probabilities"].values()) - 1) < 1e-5
    # the complete backbone, no lm_head, and the fitted temperature in config.json
    assert json.loads((out / "backbone" / "config.json").read_text())["num_hidden_layers"] == 28
    keys = set()
    for f in (out / "backbone").glob("*.safetensors"):
        with safe_open(f, "pt") as s:
            keys |= set(s.keys())
    assert {int(m.group(1)) for k in keys if (m := re.search(r"layers\.(\d+)\.", k))} == set(range(28))
    assert not any("lm_head" in k for k in keys)
    assert json.loads((out / "config.json").read_text())["temperature"] == 1.3
