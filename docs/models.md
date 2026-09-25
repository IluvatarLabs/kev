# Models

Qwen3 is the first supported backend. Conversion recipes are included for **Qwen3-1.7B, 4B, and 8B**.

**Pretrained downloads will follow on Hugging Face.** To build a model now:

```sh
NVIDIA_TF32_OVERRIDE=0 uv run python -m decision_model.convert \
  --config recipes/qwen3-1.7b.yaml --out runs/showcase
```

Use `recipes/qwen3-4b.yaml` or `recipes/qwen3-8b.yaml` for the larger parents, with a separate output directory.

The exported directory contains the complete backbone, tokenizer, decision head, calibration, and provenance.
It runs locally without the teacher:

```python
from decision_model import DecisionModel
model = DecisionModel.from_pretrained("runs/showcase/model")
```

The loader takes a local directory. Once Hub downloads are available, download the complete artifact before loading it.
