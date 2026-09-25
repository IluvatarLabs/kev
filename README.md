# Kev

**Turn your LLM into a Jev-like decision model.**

Kev is an LLM-to-decision-model conversion framework. Give the converted model a document, questions, and answer options. Get Boolean, choice, or score decisions with probabilities—without generating text. Questions and options can change at runtime; no new classifier is needed.

Convert once, then run the exported model locally without its teacher. Kev trains LoRA plus a shared pointer head, learns from option-order-corrected teacher distributions, and shares document computation across questions. The backbone stays the same size.

## Quick start

Python 3.12–3.13 and [uv](https://docs.astral.sh/uv/). CUDA is recommended for conversion.

```sh
git clone https://github.com/IluvatarLabs/kev.git
cd kev
uv sync

NVIDIA_TF32_OVERRIDE=0 uv run python -m decision_model.convert \
  --config recipes/qwen3-1.7b.yaml --out runs/showcase
```

This downloads the parent and training data, trains, calibrates, and exports `runs/showcase/model`. Recipes are also included for **Qwen3-4B** and **Qwen3-8B**.

**Pretrained downloads are coming separately on Hugging Face.** This release includes the converter, runtime, and demo; build a model with the command above to use it now.

```python
from decision_model import DecisionModel

model = DecisionModel.from_pretrained("runs/showcase/model")
result = model.evaluate(
    state="I was charged twice. Please refund the duplicate payment.",
    questions={
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want?",
            "criteria": {"refund": "Return a payment", "cancel": "End a subscription"},
        }
    },
    form="packed",
)
print(result["answers"])
```

Or use the CLI and HTTP API:

```sh
uv run python -m decision_model.decide --model runs/showcase/model --request demo/customer_support.json
uv run python -m decision_model.serve --model runs/showcase/model --port 8008
# In another terminal:
curl http://localhost:8008/v1/systemone -H 'content-type: application/json' -d @demo/customer_support.json
```

## More

- [Editable demo](demo/README.md) — change the document, questions, and options.
- [Models](docs/models.md) — supported sizes and artifact details.

Qwen3 is the first supported backend and the showcase for v0.1. The runtime supports up to 32 questions and 2,048 state tokens. Packed execution uses one pass; larger packed requests fall back to cached execution.

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
