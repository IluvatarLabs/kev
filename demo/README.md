# Editable-question demo

One customer-support email becomes five independent typed decisions: two Choice questions, two Boolean (`noul`)
questions and one ordinal Score. The converted model loads from its own directory with network access disabled. It
never calls a teacher or the parent.

```bash
python demo/run_demo.py --model runs/showcase/model --request demo/customer_support.json
# add the parent's one-token scoring (raw and cyclic) beside it; the parent must already be in the local HF cache
python demo/run_demo.py --model runs/showcase/model --parent Qwen/Qwen3-1.7B --request demo/customer_support.json
```

For every question the demo prints each option's probability under `parent/raw`, `parent/cyclic` and `converted`,
marks each column's top option with `*`, and flags any column that disagrees with the converted model. It also prints
the median warm latency per request for each system: one untimed call, then the median of three timed calls.

## Change the questions

The request is plain JSON in the same shape the Python API, CLI and HTTP endpoint accept. Edit it and rerun; nothing
is retrained.

- **Wording:** change a question's `instructions`.
- **Options:** add, remove or rename keys in a Choice `criteria` object, or edit their descriptions. A key is the
  stable id reported back, and its value is the description the model reads (`null` for none).
- **Order:** reorder a Choice `criteria` object. The probabilities stay keyed by option id, so you can see whether the
  answer depends on the order.
- **Levels:** a Score `criteria` is an ordered list of level descriptions, reported as `"0"`, `"1"`, ....
- **New questions:** add any `noul`, `choice` or `score` question under `questions`, up to 32 per request.

Ready-made variants of the same record:

| File | What changed |
|---|---|
| `customer_support.json` | the base request |
| `customer_support_reworded.json` | every question and option description reworded; same ids |
| `customer_support_reordered.json` | the two Choice questions' options in reverse order |
