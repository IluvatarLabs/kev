"""decision_model: convert a decoder LLM into a typed decision model and run it.

Public surface (kept small on purpose):
    from decision_model import DecisionModel, DecisionRequest
    m = DecisionModel.from_pretrained("runs/showcase/model")
    m.evaluate(state={...}, questions={...})

Parent identity for the V1 certified conversion. The framework takes parent/revision as arguments; this is the default.
"""
DEFAULT_PARENT = "Qwen/Qwen3-1.7B"
DEFAULT_PARENT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

from .schema import DecisionRequest, AdmissionError, MAX_OPTIONS, MAX_QUESTIONS, MAX_STATE_TOKENS, MAX_ROW_TOKENS  # noqa: E402


def __getattr__(name):
    # lazy: importing torch/transformers only when the model is actually used
    if name == "DecisionModel":
        from .model import DecisionModel
        return DecisionModel
    raise AttributeError(name)


__all__ = ["DecisionModel", "DecisionRequest", "AdmissionError", "DEFAULT_PARENT", "DEFAULT_PARENT_REVISION",
           "MAX_OPTIONS", "MAX_QUESTIONS", "MAX_STATE_TOKENS", "MAX_ROW_TOKENS"]
