"""Target identity (SPEC section 10 item 3): permuted teacher layouts must map back to the right semantic option before
the log-mean, and training augmentation must keep each target weight on its option. An off-by-one here trains a model
on correctly shaped but wrong labels."""
import hashlib
import random
import re

import numpy as np
import pytest

from decision_model import DEFAULT_PARENT, DEFAULT_PARENT_REVISION
from decision_model.render import to_record
from decision_model.schema import DecisionRequest
from decision_model.targets import ParentScorer
from decision_model.train import example

OPTION_LINE = re.compile(r"^([A-Z]|\d+)\. (.*)$", re.M)


def f(text):
    """Fixed per-option logit, a function of the option's identity only."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:6], 16) / 0xFFFFFF * 4.0


def fake_forward(tok, bias):
    """Logit at position j = f(option shown at j) + bias * j (an additive position bias); label log-probs returned with
    a full-vocabulary mass of 0.8, as a real parent would leave some mass off the labels."""
    def forward(prompts, label_ids):
        out = []
        for prompt, ids in zip(prompts, label_ids):
            shown = [m.group(2) for m in OPTION_LINE.finditer(prompt)]
            if not shown:   # noul: the labels are the options themselves
                shown = [tok.decode([i]).strip() for i in ids]
            z = np.array([f(s) + bias * j for j, s in enumerate(shown)])
            out.append(z - np.log(np.exp(z).sum()) + np.log(0.8))
        return out
    return forward


@pytest.fixture(scope="module")
def scorer():
    return ParentScorer(DEFAULT_PARENT, DEFAULT_PARENT_REVISION, device="cpu", load_model=False)


REQ = {"state": {"message": "The Lakers beat the Celtics 102-99 in overtime."},
       "questions": {"topic": {"type": "choice", "instructions": "What is the topic?",
                               "criteria": {"world": "World news", "sports": None, "business": "Markets and companies", "scitech": "Science", "arts": None}},
                     "is_sports": {"type": "noul", "instructions": "Is it about sports?"},
                     "tone": {"type": "score", "instructions": "How excited is the writer?", "criteria": ["calm", "engaged", "thrilled"]}}}


def test_cyclic_layouts_map_back_to_semantic_options_through_augmentation(scorer):
    req = DecisionRequest.model_validate(REQ)
    rec, meta = to_record(req)
    shown = {m["id"]: dict(zip(m["keys"], q["options"])) for q, m in zip(rec["questions"], meta)}
    expected = {}
    for m in meta:
        texts = shown[m["id"]] if m["type"] != "noul" else {"true": "Yes", "false": "No"}
        z = np.array([f(texts[k]) for k in m["keys"]])
        expected[m["id"]] = dict(zip(m["keys"], np.exp(z) / np.exp(z).sum()))

    # no position bias: every layout, mapped back by key, is the same vector, and the log-mean equals it
    scorer.forward = fake_forward(scorer.tok, bias=0.0)
    det = scorer.score_detail([req], "cyclic")[0]
    assert [det[q]["layouts_count"] for q in ("topic", "is_sports", "tone")] == [5, 2, 1]
    for qid, d in det.items():
        for lay in d["layouts"]:
            lp = np.array(lay["logprobs_by_position"])
            p = np.exp(lp) / np.exp(lp).sum()
            by_key = {d["keys"][i]: p[j] for j, i in enumerate(lay["perm"])}
            assert all(abs(by_key[k] - expected[qid][k]) < 1e-12 for k in d["keys"])
        assert all(abs(d["cyclic"][k] - expected[qid][k]) < 1e-12 for k in d["keys"])

    # additive position bias: raw is distorted, the cyclic log-mean cancels it exactly (choice has every rotation)
    scorer.forward = fake_forward(scorer.tok, bias=0.7)
    det = scorer.score_detail([req], "cyclic")[0]
    topic = det["topic"]
    assert max(abs(topic["raw"][k] - expected["topic"][k]) for k in topic["keys"]) > 0.05
    assert all(abs(topic["cyclic"][k] - expected["topic"][k]) < 1e-12 for k in topic["keys"])

    # training augmentation reorders options; every target weight must stay on its own option text
    record = {"record_id": "r0", "request": REQ,
              "supervision": {"topic": {"gold": "sports"}, "is_sports": {"gold": "true"}, "tone": {"gold": "2"}}}
    teacher = {("r0", qid): d["cyclic"] for qid, d in det.items()}
    orders = set()
    for seed in range(6):
        areq, vecs = example(record, teacher, random.Random(seed), gold_weight=0.5)
        arec, ameta = to_record(areq)
        for q, m, t in zip(arec["questions"], ameta, vecs):
            gold = record["supervision"][m["id"]]["gold"]
            for key, text, w in zip(m["keys"], q["options"], t):
                assert text == shown[m["id"]][key]
                assert abs(w - (0.5 * (key == gold) + 0.5 * teacher[("r0", m["id"])][key])) < 1e-12
            if m["type"] != "choice":
                assert m["keys"] == [x["keys"] for x in meta if x["id"] == m["id"]][0]   # noul/score never reordered
        orders.add(tuple(ameta[0]["keys"]))
    assert len(orders) > 1

