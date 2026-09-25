"""Public request/result contract (lead-owned). Every entry point (Python API, CLI, HTTP, trainer, evaluator, benchmark)
validates requests with DecisionRequest and reports probabilities under the keys question_keys() defines.

Request (Jev-shaped):
    {"state": <JSONContent>,
     "questions": {qid: {"type": "noul"|"choice"|"score", "instructions": <JSONContent>, "criteria": ...}}}
    noul   criteria: optional {"true": desc, "false": desc}
    choice criteria: {option_id: desc|None}  (1..255, insertion order = canonical presentation order)
    score  criteria: [level_desc, ...]       (1..255, ordered)

Result per question (complete distribution, no rounding, keyed by stable option ID):
    noul   {"type": "noul", "p_true": p, "decision": p >= 0.5, "probabilities": {"false": 1-p, "true": p}}
    choice {"type": "choice", "choice": argmax_id, "top_probability": max(p), "probabilities": {id: p}}
    score  {"type": "score", "level": modal_index, "expected_level": sum(i*p_i), "top_probability": p[mode],
            "levels": {"0": desc, ...}, "probabilities": {"0": p0, ...}}
`top_probability` is the probability of the modal option, not the probability of being correct.

Admission limits (V1 defaults, engineering limits to validate, not the parent's full context):
    MAX_STATE_TOKENS 2048 (state incl. its delimiter), MAX_ROW_TOKENS 4096 (state + one question branch),
    MAX_QUESTIONS 32 per request, MAX_OPTIONS 255. Over-limit requests raise AdmissionError; nothing is ever truncated.
"""
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255
MAX_QUESTIONS = 32
MAX_STATE_TOKENS = 2048
MAX_ROW_TOKENS = 4096


class AdmissionError(ValueError):
    """The request exceeds an admission limit (state tokens, row tokens, questions, options). Serving maps it to 422."""


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.criteria is not None and not set(self.criteria) <= {"true", "false"}:
            raise ValueError("noul criteria keys must be 'true' and/or 'false'")
        return self


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"choice criteria must have 1..{MAX_OPTIONS} options")
        if any(not k for k in self.criteria):
            raise ValueError("choice option ids must be non-empty strings")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


Question = Union[Noul, Choice, Score]


class DecisionRequest(BaseModel):
    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)
    model: str = "decision-model"   # Jev-shaped field; ignored by the local runtime, echoed in responses


def question_keys(qtype: str, criteria) -> list[str]:
    """Stable IDs the probabilities are reported under, in canonical option order: the criteria ids (choice),
    ["false", "true"] (noul), the level indices as strings (score). Labels, targets and results all use these."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_answers(probs: list[list[float]], meta: list[dict]) -> dict[str, dict[str, Any]]:
    """Public result objects from per-question probability vectors (canonical option order) and the per-question
    metadata render.to_record() returns ({"id", "type", "keys", "legend" for score})."""
    out = {}
    for p, m in zip(probs, meta):
        p = [float(x) for x in p]
        if len(p) != len(m["keys"]):
            raise ValueError(f"question {m['id']}: {len(p)} probabilities for {len(m['keys'])} options")
        mode = max(range(len(p)), key=lambda i: p[i])
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "p_true": p[1], "decision": p[1] >= 0.5, "probabilities": dict(zip(m["keys"], p))}
        elif m["type"] == "choice":
            out[m["id"]] = {"type": "choice", "choice": m["keys"][mode], "top_probability": p[mode], "probabilities": dict(zip(m["keys"], p))}
        else:
            out[m["id"]] = {"type": "score", "level": mode, "expected_level": sum(i * pi for i, pi in enumerate(p)),
                            "top_probability": p[mode], "levels": m["legend"], "probabilities": dict(zip(m["keys"], p))}
    return out
