# Modified implementation for Kev; see LICENSE and NOTICE.
"""Prepare the default conversion corpus: `python -m decision_model.data --config recipes/qwen3-1.7b.yaml`.

Writes data/prepared/{train,dev,calibration,test,showcase,unseen}.jsonl and manifest.json. One record per line:

    {"record_id", "group_id", "source_family", "schema_family", "split",
     "request": <DecisionRequest JSON>, "supervision": {qid: {"gold": <stable key>, "task": str}}, "provenance": {...}}

Gold labels use the stable keys schema.question_keys() defines: the option id (choice), "false"/"true" (noul), the level
index as a string (score).

Split discipline (SPEC section 6): official test/validation rows go only to `test` and `showcase`; the train-split pool
of each public source is filtered to the training context, grouped by normalized text, and split 70/10/10/10 by a hash
of the group id into train/dev/calibration/newly-reserved test before train is capped. Rows whose text also occurs in
the source's official evaluation split are excluded from the pool. Executable policy states come from Kev's generators;
whole rule structures (Kev DEV_SHAPES / TEST_SHAPES and one contrastive family) are held out for transfer panels.
PAWS, QNLI and Emotion form the eval-only `unseen` panel.
"""
import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import yaml

from .render import encode, render_content, to_record
from .schema import AdmissionError, DecisionRequest, question_keys

SPLITS = ("train", "dev", "calibration", "test", "showcase", "unseen")
ENCODING = "utf-8"


# --- files and hashes (kev/suite.py) -----------------------------------------------------------------------------------

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(" ".join(str(text).casefold().split()).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding=ENCODING).splitlines() if line.strip()]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding=ENCODING)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding=ENCODING)


def load_split(directory, split):
    """Records of one prepared split, verified against the manifest's sha256 and count."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding=ENCODING))
    path = directory / f"{split}.jsonl"
    entry = manifest["files"][path.name]
    if digest(path) != entry["sha256"]:
        raise ValueError(f"prepared data checksum mismatch: {path}")
    records = read_jsonl(path)
    if len(records) != entry["records"]:
        raise ValueError(f"prepared data record count mismatch: {path}")
    return records


def request_of(record) -> DecisionRequest:
    return DecisionRequest.model_validate(record["request"])


def source_seed(seed, source):
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big")


def bucket(group_id, seed):
    """Deterministic 70/10/10/10 group assignment, independent of pool composition."""
    b = int.from_bytes(hashlib.sha256(f"{seed}:split:{group_id}".encode()).digest()[:8], "big") % 10
    return "train" if b < 7 else "dev" if b == 7 else "calibration" if b == 8 else "test"


# --- public-source converters (kev/data.py) ----------------------------------------------------------------------------
# A converter maps (row, rng, variant) to a labelled request {"state", "questions": {qid: {type, instructions, criteria,
# "label", "src"}}} with label = option id (choice), bool (noul) or level index (score). variant=True produces the
# showcase's changed schema for the same row: new wording, new option descriptions, a different option order.

AG = {"world": "World news: politics, international affairs, conflicts", "sports": "Sports: games, athletes, teams, results",
      "business": "Business: companies, markets, economy, finance", "scitech": "Science and technology: research, gadgets, software, space"}
AG_VARIANT = {"sports": "Athletics, matches and tournaments", "scitech": "Research, computing, gadgets and space",
              "business": "Companies, earnings and markets", "world": "International politics, diplomacy and conflict"}
MNLI = {"entailment": "The hypothesis follows from the premise", "neutral": "The hypothesis may or may not be true given the premise",
        "contradiction": "The hypothesis contradicts the premise"}
MNLI_VARIANT = {"contradiction": "The premise rules the hypothesis out", "entailment": "The premise guarantees the hypothesis",
                "neutral": "The premise neither guarantees nor rules out the hypothesis"}
SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]
SST5_VARIANT = ["strongly negative opinion", "somewhat negative opinion", "mixed or no opinion", "somewhat positive opinion", "strongly positive opinion"]
TREC = {"abbreviation": "Asks what an abbreviation stands for", "entity": "Asks about a thing, object, animal, product, or creative work",
        "description": "Asks for a definition, description, reason, or manner", "human": "Asks about a person, group, or organisation",
        "location": "Asks about a place", "number": "Asks for a number, date, count, or other numeric value"}
TREC_VARIANT = {"number": "A quantity, date or measurement", "location": "A place", "human": "A person or organisation",
                "entity": "A thing, product, animal or work", "description": "An explanation or definition", "abbreviation": "The expansion of an abbreviation"}
EMOTION = {"sadness": None, "joy": None, "love": None, "anger": None, "fear": None, "surprise": None}
BANK_TEMPLATES = ["Customer asks about {}", "Issue concerning {}", "Request related to {}", "{}"]


def _wrap_state(text, rng):
    r = rng.random()
    if r < 0.15: return {"document": text}
    if r < 0.25: return {"ticket": {"channel": rng.choice(["email", "chat", "web form"]), "body": text}}
    if r < 0.32: return [{"role": "customer", "content": text}]
    return text


def _instr(text, rng):
    return {"question": text, "focus": rng.choice(["Use only the information given.", "Pick the single best fit.", "Consider the whole message."])} if rng.random() < 0.15 else text


def _desc(desc, rng, p_null=0.3, p_struct=0.1):
    r = rng.random()
    if r < p_null: return None
    if r < p_null + p_struct: return {"what": desc}
    return desc


def _banking(row, rng, names, variant=False):
    if variant:
        crit = {k: None for k in reversed(names)}
        return {"state": {"customer_message": row["text"]}, "questions": {"intent": {"type": "choice", "instructions": "What does the customer want help with?",
                "criteria": crit, "label": names[row["label"]], "src": "banking77"}}}
    t = rng.choice(BANK_TEMPLATES)
    crit = {k: _desc(t.format(k.replace("_", " ")), rng, p_null=0.5, p_struct=0.0) for k in names}
    return {"state": _wrap_state(row["text"], rng), "questions": {"intent": {"type": "choice", "instructions": _instr("Which banking intent best describes this customer message?", rng),
            "criteria": crit, "label": names[row["label"]], "src": "banking77"}}}


def _boolq(row, rng, names=None, variant=False):
    question = row["question"].strip().rstrip("?") + "?"
    if variant:
        return {"state": {"passage": row["passage"]}, "questions": {"answer": {"type": "noul", "instructions": f"According to the passage: {question}",
                "criteria": {"true": "The passage confirms it", "false": "The passage denies it or does not say"}, "label": bool(row["answer"]), "src": "boolq"}}}
    q = {"type": "noul", "instructions": _instr(question, rng), "label": bool(row["answer"]), "src": "boolq"}
    if rng.random() < 0.4: q["criteria"] = {"true": "The passage supports a yes answer", "false": "The passage supports a no answer or does not say"}
    return {"state": _wrap_state(row["passage"], rng), "questions": {"answer": q}}


def _agnews(row, rng, names=None, variant=False):
    keys = list(AG)
    y = keys[row["label"]]
    if variant:
        return {"state": {"article": row["text"]}, "questions": {"topic": {"type": "choice", "instructions": "Which newspaper section would this story run in?",
                "criteria": dict(AG_VARIANT), "label": y, "src": "agnews"}}}
    qs = {"topic": {"type": "choice", "instructions": _instr("What is the topic of this article?", rng), "criteria": {k: _desc(v, rng) for k, v in AG.items()}, "label": y, "src": "agnews"}}
    for k in rng.sample(keys, 2):
        qs[f"is_{k}"] = {"type": "noul", "instructions": f"Is this article about {AG[k].split(':')[0].lower()}?", "label": k == y, "src": "agnews_yn"}
    return {"state": _wrap_state(row["text"], rng), "questions": qs}


def _mnli(row, rng, names=None, variant=False):
    keys = list(MNLI)
    if variant:
        return {"state": {"premise": row["premise"]}, "questions": {"relation": {"type": "choice",
                "instructions": f'Does the premise support, rule out, or leave open this hypothesis: "{row["hypothesis"]}"?',
                "criteria": dict(MNLI_VARIANT), "label": keys[row["label"]], "src": "mnli"}}}
    return {"state": _wrap_state(row["premise"], rng), "questions": {"relation": {"type": "choice", "instructions": _instr(f'Hypothesis: "{row["hypothesis"]}" How does it relate to the premise?', rng),
            "criteria": {k: _desc(v, rng) for k, v in MNLI.items()}, "label": keys[row["label"]], "src": "mnli"}}}


def _sst5(row, rng, names=None, variant=False):
    if variant:
        return {"state": {"review": row["text"]}, "questions": {"sentiment": {"type": "score", "instructions": "How positive or negative is the writer's opinion?",
                "criteria": list(SST5_VARIANT), "label": row["label"], "src": "sst5"}}}
    return {"state": _wrap_state(row["text"], rng), "questions": {"sentiment": {"type": "score", "instructions": _instr("What is the sentiment of this review sentence?", rng),
            "criteria": list(SST5), "label": row["label"], "src": "sst5"}}}


def _trec(row, rng, names=None, variant=False):
    keys = list(TREC)
    if variant:
        return {"state": {"question": row["text"]}, "questions": {"answer_type": {"type": "choice", "instructions": "What type of information would a correct answer provide?",
                "criteria": dict(TREC_VARIANT), "label": keys[row["coarse_label"]], "src": "trec"}}}
    return {"state": _wrap_state(row["text"], rng), "questions": {"answer_type": {"type": "choice", "instructions": "What kind of answer does this question ask for?",
            "criteria": dict(TREC), "label": keys[row["coarse_label"]], "src": "trec"}}}


def _emotion(row, rng, names=None, variant=False):
    keys = list(EMOTION)
    return {"state": row["text"], "questions": {"emotion": {"type": "choice", "instructions": "Which emotion does the writer express?",
            "criteria": dict(EMOTION), "label": keys[row["label"]], "src": "emotion"}}}


def _qnli(row, rng, names=None, variant=False):
    return {"state": _wrap_state(row["sentence"], rng), "questions": {"answers": {"type": "noul", "instructions": f'Does the sentence contain the answer to this question: "{row["question"]}"',
            "label": row["label"] == 0, "src": "qnli"}}}


def _paws(row, rng, names=None, variant=False):
    return {"state": _wrap_state(row["sentence1"], rng), "questions": {"paraphrase": {"type": "noul", "instructions": f'Does this sentence mean the same thing: "{row["sentence2"]}"',
            "criteria": {"true": "Same meaning, possibly reworded", "false": "Different meaning, even if most words match"}, "label": row["label"] == 1, "src": "paws"}}}


AMAZON = ["1 star: very negative", "2 stars: negative", "3 stars: mixed", "4 stars: positive", "5 stars: very positive"]


def _mcq(question, labels, texts, answer_label, src, rng):
    """Kev _mcq: knowledge MCQ -> Choice with neutral keys opt_1..opt_K; option order shuffled per record."""
    order = list(range(len(texts))); rng.shuffle(order)
    keys = [f"opt_{i + 1}" for i in range(len(texts))]
    crit = {keys[i]: texts[j] for i, j in enumerate(order)}
    return {"state": {"question": question}, "questions": {"answer": {"type": "choice", "instructions": "Which option correctly answers the question?",
            "criteria": crit, "label": keys[order.index(labels.index(answer_label))], "src": src}}}


def _mmlu(row, rng, names=None, variant=False):
    keys = ["a", "b", "c", "d"]
    return {"state": {"subject": row["subject"].replace("_", " "), "question": row["question"]},
            "questions": {"answer": {"type": "choice", "instructions": "Which option correctly answers the question?",
                                     "criteria": dict(zip(keys, row["choices"])), "label": keys[row["answer"]], "src": "mmlu"}}}


def _arc(row, rng, names=None, variant=False):
    return _mcq(row["question"], list(row["choices"]["label"]), list(row["choices"]["text"]), row["answerKey"], "arc", rng)


def _sciq(row, rng, names=None, variant=False):
    options = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
    keys = ["a", "b", "c", "d"]; order = list(range(4)); rng.shuffle(order)
    crit = {keys[i]: options[j] for i, j in enumerate(order)}
    return {"state": {"passage": row["support"], "question": row["question"]} if row["support"] else {"question": row["question"]},
            "questions": {"answer": {"type": "choice", "instructions": "Which option answers the science question?", "criteria": crit,
                                     "label": keys[order.index(0)], "src": "sciq"}}}


def _offensive(row, rng, names=None, variant=False):
    return {"state": row["text"], "questions": {"offensive": {"type": "noul", "instructions": "Is this post offensive?",
            "criteria": {"true": "Contains insults, threats, profanity directed at someone, or hateful content", "false": "Not offensive"},
            "label": row["label"] == 1, "src": "tweet_offensive"}}}


def _imdb(row, rng, names=None, variant=False):
    return {"state": _wrap_state(" ".join(row["text"].replace("<br />", " ").split()[:220]), rng), "questions": {"positive": {"type": "noul",
            "instructions": "Is this movie review positive?", "criteria": {"true": "The reviewer liked the film overall", "false": "The reviewer disliked the film overall"},
            "label": row["label"] == 1, "src": "imdb"}}}


def _amazon(row, rng, names=None, variant=False):
    return {"state": _wrap_state(" ".join(row["text"].split()[:220]), rng), "questions": {"stars": {"type": "score",
            "instructions": "How many stars did this product reviewer give?", "criteria": list(AMAZON), "label": row["label"], "src": "amazon"}}}


def _dbpedia(row, rng, names, variant=False):
    keys = [x.lower().replace(" ", "_") for x in names]
    return {"state": _wrap_state(" ".join(row["content"].split()[:200]), rng), "questions": {"category": {"type": "choice",
            "instructions": "Which category does the subject of this encyclopedia text belong to?", "criteria": {k: None for k in keys},
            "label": keys[row["label"]], "src": "dbpedia14"}}}


# extended never-trained panel (evaluation only; written by build_panel, never by prepare)
EXTENDED = {
    "mmlu": ("cais/mmlu", "all", None, None, "test", _mmlu, "mmlu.answer", "question"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", None, None, "test", _arc, "arc.answer", "question"),
    "sciq": ("allenai/sciq", None, None, None, "test", _sciq, "sciq.answer", "question"),
    "tweet_offensive": ("cardiffnlp/tweet_eval", "offensive", None, None, "test", _offensive, "tweet_offensive.offensive", "text"),
    "imdb": ("stanfordnlp/imdb", None, None, None, "test", _imdb, "imdb.positive", "text"),
    "amazon": ("SetFit/amazon_reviews_multi_en", None, None, None, "test", _amazon, "amazon.stars", "text"),
    "dbpedia14": ("fancyzhx/dbpedia_14", None, None, None, "test", _dbpedia, "dbpedia14.category", "content"),
}


# name: (repo, config, branch to pin, train split, official evaluation split, converter, schema family, grouping text field)
PUBLIC = {
    "agnews": ("fancyzhx/ag_news", None, None, "train", "test", _agnews, "agnews.topic", "text"),
    "trec": ("CogComp/trec", None, "refs/convert/parquet", "train", "test", _trec, "trec.answer_type", "text"),
    "banking77": ("legacy-datasets/banking77", None, None, "train", "test", _banking, "banking77.intent", "text"),
    "boolq": ("google/boolq", None, None, "train", "validation", _boolq, "boolq.answer", "passage"),
    "mnli": ("nyu-mll/multi_nli", None, None, "train", "validation_matched", _mnli, "mnli.relation", "premise"),
    "sst5": ("SetFit/sst5", None, None, "train", "test", _sst5, "sst5.sentiment", "text"),
}
UNSEEN = {
    "paws": ("google-research-datasets/paws", "labeled_final", None, None, "test", _paws, "paws.paraphrase", "sentence1"),
    "qnli": ("nyu-mll/glue", "qnli", None, None, "validation", _qnli, "qnli.answers", "sentence"),
    "emotion": ("dair-ai/emotion", "split", None, None, "test", _emotion, "emotion.emotion", "text"),
}


def _label_ok(row):
    return row.get("label", 0) != -1


# --- executable policy generators (kev/composition.py, kev/contrastive.py) ---------------------------------------------

UNKNOWN = None
SHAPES = {
    "atom": 0, "negation": ("not", 0), "conjunction": ("and", 0, 1), "disjunction": ("or", 0, 1), "exception": ("unless", 0, 1),
    "conditional": ("if", 0, 1, 2), "nested_and": ("and", ("and", 0, 1), 2), "nested_or": ("or", 0, ("or", 1, 2)),
    "held_and_or": ("and", ("or", 0, 1), 2), "held_or_not": ("or", ("and", 0, 1), ("not", 2)), "held_conditional": ("if", 0, ("not", 1), 2),
    "final_combination": ("and", ("if", 0, 1, 2), 3), "final_negation": ("not", ("or", ("and", 0, 1), 2)), "final_exception": ("or", ("unless", 0, 1), 2),
}
TRAIN_SHAPES = tuple(list(SHAPES)[:8])
DEV_SHAPES = tuple(list(SHAPES)[8:11])
TEST_SHAPES = tuple(list(SHAPES)[11:])
KINDS = ("lt", "le", "gt", "ge", "eq", "range", "match", "elapsed", "flag")


def atom_value(atom, facts):
    if any(key not in facts for key in atom["fields"]):
        return UNKNOWN
    values = [facts[k] for k in atom["fields"]]
    kind, threshold = atom["kind"], atom["threshold"]
    x = values[0]
    if kind == "lt": return x < threshold
    if kind == "le": return x <= threshold
    if kind == "gt": return x > threshold
    if kind == "ge": return x >= threshold
    if kind == "eq": return x == threshold
    if kind == "range": return threshold <= x <= threshold + 10
    if kind == "match": return x == values[1]
    if kind == "elapsed": return (date.fromisoformat(values[1]) - date.fromisoformat(x)).days <= threshold
    if kind == "flag": return x
    raise ValueError(f"unknown atom {kind}")


def evaluate_rule(tree, atoms, facts):
    if isinstance(tree, int):
        return atom_value(atoms[tree], facts)
    op, *children = tree
    values = [evaluate_rule(t, atoms, facts) for t in children]
    if op == "not": return None if values[0] is None else not values[0]
    if op == "unless":
        a, exception = values
        values = [a, None if exception is None else not exception]
        op = "and"
    if op == "and":
        return False if False in values else None if None in values else True
    if op == "or":
        return True if True in values else None if None in values else False
    if op == "if":
        condition, yes, no = values
        return yes if condition is True else no if condition is False else yes if yes == no else None
    raise ValueError(f"unknown operation {op}")


def atom_text(atom):
    kind, fields, t = atom["kind"], atom["fields"], atom["threshold"]
    key = fields[0]
    words = {"lt": "less than", "le": "at most", "gt": "greater than", "ge": "at least", "eq": "equal to"}
    if kind in words: return f"{key} is {words[kind]} {t}"
    if kind == "range": return f"{key} is between {t} and {t + 10}, including both endpoints"
    if kind == "match": return f"{fields[0]} is the same person as {fields[1]}"
    if kind == "elapsed": return f"the elapsed days from {fields[0]} to {fields[1]} are at most {t}"
    return f"{key} is yes"


def render_rule(tree, atoms, style):
    """Kev's surface styles: 0 logic-like, 1 'both/at least one of' lists, 2 reserved for the locked test, 3 prose, 4 clause per line."""
    if isinstance(tree, int): return atom_text(atoms[tree])
    op, *children = tree
    parts = [render_rule(c, atoms, style) for c in children]
    if style == 3:
        if op == "not": return f"the condition \"{parts[0]}\" fails"
        if op == "and": return f"{parts[0]}, and also {parts[1]}"
        if op == "or": return f"either {parts[0]}, or else {parts[1]}"
        if op == "unless": return f"{parts[0]}, except when {parts[1]}"
        return f"when {parts[0]} the requirement is that {parts[1]}, and when it is not the requirement is that {parts[2]}"
    if style == 4:
        if op == "not": return f"[NOT: {parts[0]}]"
        if op in ("and", "or"): return f"[{'ALL' if op == 'and' else 'ANY'} of: {parts[0]} | {parts[1]}]"
        if op == "unless": return f"[{parts[0]} UNLESS {parts[1]}]"
        return f"[IF {parts[0]} THEN {parts[1]} ELSE {parts[2]}]"
    if op == "not": return f"NOT ({parts[0]})" if style == 0 else f"it is not the case that ({parts[0]})"
    if op in ("and", "or"):
        if style == 0: return f"({parts[0]}) {op.upper()} ({parts[1]})"
        connector = "both" if op == "and" else "at least one of"
        return f"{connector} these conditions hold: [({parts[0]}); ({parts[1]})]"
    if op == "unless": return f"({parts[0]}) holds and the exception ({parts[1]}) does not hold"
    return f"if ({parts[0]}), use ({parts[1]}); otherwise use ({parts[2]})"


def leaf_indices(tree):
    if isinstance(tree, int): return {tree}
    return set().union(*(leaf_indices(t) for t in tree[1:]))


def make_atoms(tree, rng):
    nouns = rng.sample(["request", "account", "package", "review", "member", "shipment", "entry", "case"], 4)
    atoms = []
    for i in range(max(leaf_indices(tree)) + 1):
        kind = rng.choice(KINDS)
        prefix = nouns[i]
        fields = [f"{prefix} value"]
        if kind == "match": fields = [f"{prefix} signer", f"{prefix} designated approver"]
        elif kind == "elapsed": fields = [f"{prefix} start date", f"{prefix} end date"]
        elif kind == "flag": fields = [f"{prefix} verified"]
        atoms.append({"kind": kind, "fields": fields, "threshold": rng.randint(5, 60)})
    return atoms


def fact_domains(atoms, rng):
    domains = {}
    for a in atoms:
        kind, fs, t = a["kind"], a["fields"], a["threshold"]
        if kind == "match":
            people = rng.sample(["Mira", "Noah", "Aiko", "Ravi", "Sana", "Elin", "Tomas", "Kofi"], 3)
            domains.update({k: people for k in fs})
        elif kind == "elapsed":
            day = date(2027, rng.randint(1, 8), rng.randint(1, 28))
            domains[fs[0]] = [day.isoformat()]
            domains[fs[1]] = [(day + timedelta(days=n)).isoformat() for n in (max(0, t - 1), t, t + 1, t + 10)]
        elif kind == "flag": domains[fs[0]] = [False, True]
        else: domains[fs[0]] = [t - 1, t, t + 1, t + 10, t + 11]
    domains["routing reference"] = [rng.randint(100, 500), rng.randint(501, 999)]
    return domains


def rendered_facts(facts, order):
    def value(v):
        return "yes" if v is True else "no" if v is False else str(v)
    return [f"The {k} is {value(facts[k])}." for k in order]


POLICY_WRAPPERS = {
    0: "Approve exactly when {rule}. Otherwise deny. The routing reference does not affect eligibility.",
    1: "Approve exactly when {rule}. Otherwise deny. The routing reference does not affect eligibility.",
    2: "Approval requires the following rule to be true: {rule}. A false rule means denial. Routing references are irrelevant.",
    3: "A case is approved when {rule}; any other case is denied. Routing references play no part in the decision.",
    4: "DECISION RULE {rule} -> approve; otherwise reject. Ignore the routing reference.",
}


def composition_generate(groups_per_shape, seed, shapes, styles=(0, 1)):
    """Kev composition.generate: per group, a relevant pair (one deciding fact flips the answer) and an irrelevant pair
    (the routing reference changes, the answer does not); four states sharing one group."""
    records = []
    for shape in shapes:
        tree = SHAPES[shape]
        rng = random.Random(f"{seed}:{shape}")
        for i in range(groups_per_shape):
            atoms = make_atoms(tree, rng)
            domains = fact_domains(atoms, rng)
            changed = None
            for _attempt in range(1000):
                facts = {k: rng.choice(v) for k, v in domains.items()}
                label = evaluate_rule(tree, atoms, facts)
                keys = list(domains); rng.shuffle(keys)
                for key in keys:
                    for value in domains[key]:
                        edited = {**facts, key: value}
                        if evaluate_rule(tree, atoms, edited) != label:
                            missing = {k: v for k, v in facts.items() if k != key}
                            if evaluate_rule(tree, atoms, missing) is None:
                                changed = (edited, key)
                                break
                    if changed: break
                if changed: break
            if changed is None:
                raise ValueError(f"cannot create a decisive edit for {shape}")
            edited, deciding = changed
            nuisance = {**facts, "routing reference": next(v for v in domains["routing reference"] if v != facts["routing reference"])}
            order = list(facts); rng.shuffle(order)
            style = rng.choice(styles)
            policy = POLICY_WRAPPERS[style].format(rule=render_rule(tree, atoms, style))
            keys = ["accept", "reject"]; rng.shuffle(keys)
            criteria = {k: "The policy permits this case" if k == "accept" else "The policy does not permit this case" for k in keys}
            group = f"composition/{seed}/{shape}/{i}"
            for kind, a, b in (("relevant", facts, edited), ("irrelevant", facts, nuisance)):
                for sibling, values in (("a", a), ("b", b)):
                    result = evaluate_rule(tree, atoms, values)
                    state = {"policy": policy, "case": " ".join(rendered_facts(values, order))}
                    records.append({"state": state, "questions": {"decision": {"type": "choice", "instructions": "Apply the policy to this case.",
                                    "criteria": dict(criteria), "label": "accept" if result else "reject", "src": f"composition_{shape}"}},
                                    "_meta": {"id": f"{group}/{kind}/{sibling}", "group_id": group, "source": "composition", "family": shape,
                                              "pair_kind": kind, "sibling": sibling, "render_style": style,
                                              "certificate": {"tree": tree, "atoms": atoms, "facts": values, "order": order, "deciding_field": deciding, "label": result}}})
    return records


UNDETERMINED = "UNDETERMINED"
NAMES = ["Mira", "Noah", "Priya", "Tomas", "Aiko", "Lena", "Omar", "Sana", "Jonas", "Ravi", "Elin", "Kofi"]
ROLES = ["account owner", "billing manager", "support agent", "warehouse lead"]
ITEMS = ["a pair of running shoes", "a desk lamp", "a wireless keyboard", "a rain jacket", "a coffee grinder", "a backpack"]
PROGRAMS = ["the volunteer driver program", "the apprenticeship", "the rental agreement", "the night-shift roster"]


def _day(d):
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _need(facts, *keys):
    return all(k in facts for k in keys)


def family_return_window(rng):
    window = rng.choice([14, 30, 45, 60]); item = rng.choice(ITEMS); name = rng.choice(NAMES)
    bought = date(2026, rng.randint(1, 9), rng.randint(1, 28))
    def evaluate(f):
        return (f["request"] - f["purchase"]).days <= window if _need(f, "request", "purchase") else UNDETERMINED
    def build(days):
        request = bought + timedelta(days=days)
        return {"policy": f"Returns are accepted only if the return request is submitted within {window} days of the purchase date.",
                "sentences": [(f"{name} bought {item} on {_day(bought)}.", {"purchase": bought}),
                              (f"The return request was submitted on {_day(request)}.", {"request": request}),
                              (f"The order was paid by card and shipped to {name}'s home address.", {})],
                "evaluate": evaluate, "question": {"type": "noul", "instructions": "Is this return request within the policy window?"}}
    return build(rng.randint(1, window - 1)), build(window + rng.randint(1, 30))


def family_spend_threshold(rng):
    limit = rng.choice([250, 500, 1000, 2500]); name = rng.choice(NAMES); role = rng.choice(ROLES)
    def evaluate(f):
        return ("auto_approved" if f["amount"] <= limit else "director_signoff") if _need(f, "amount") else UNDETERMINED
    def build(amount):
        return {"policy": f"Expense claims of ${limit:,} or less are approved automatically. Claims above ${limit:,} require director sign-off.",
                "sentences": [(f"{name}, the {role}, submitted an expense claim.", {}),
                              (f"The claim total is ${amount:,}.", {"amount": amount}),
                              ("Receipts were attached for every line item.", {})],
                "evaluate": evaluate, "question": {"type": "choice", "instructions": "How is this claim handled under the policy?",
                                                   "criteria": {"auto_approved": "Approved without further review", "director_signoff": "Requires director sign-off", "rejected": "Rejected outright"}}}
    return build(limit - rng.randint(1, limit // 2)), build(limit + rng.randint(1, limit))


def family_authorization(rng):
    approver, other = rng.sample(NAMES, 2); account = rng.randint(10, 99); amount = rng.choice([40, 120, 350, 900])
    def evaluate(f):
        return f["signer"] == f["approver"] if _need(f, "signer", "approver") else UNDETERMINED
    def build(signer):
        return {"policy": "A refund is authorized only when its sole authorization was signed by someone who may authorize refunds for that account.",
                "sentences": [(f"Only {approver} may authorize refunds for account {account}.", {"approver": approver}),
                              (f"The sole authorization for this refund on account {account} was signed by {signer}.", {"signer": signer}),
                              (f"The refund amount is ${amount}.", {})],
                "evaluate": evaluate, "question": {"type": "noul", "instructions": "Is the refund authorized?"}}
    return build(approver), build(other)


def family_age_eligibility(rng):
    minimum = rng.choice([16, 18, 21, 25]); name = rng.choice(NAMES); program = rng.choice(PROGRAMS)
    def evaluate(f):
        return f["age"] >= minimum if _need(f, "age") else UNDETERMINED
    def build(age):
        return {"policy": f"Applicants must be at least {minimum} years old to be eligible for {program}.",
                "sentences": [(f"{name} applied to join {program}.", {}),
                              (f"{name} is {age} years old.", {"age": age}),
                              ("The application form was complete and signed.", {})],
                "evaluate": evaluate, "question": {"type": "noul", "instructions": "Is the applicant eligible?"}}
    return build(minimum + rng.randint(0, 20)), build(minimum - rng.randint(1, 5))


def family_quantity_limit(rng):
    limit = rng.choice([2, 3, 5, 10]); item = rng.choice(ITEMS); name = rng.choice(NAMES)
    def evaluate(f):
        if not _need(f, "qty"): return UNDETERMINED
        return "within_limit" if f["qty"] <= limit else "slightly_over" if f["qty"] <= 2 * limit else "far_over"
    def build(qty):
        return {"policy": f"Customers may order at most {limit} units of any single item per order. Orders up to double the limit are held for review; larger orders are cancelled.",
                "sentences": [(f"{name} placed an order for {item}.", {}),
                              (f"The order quantity is {qty}.", {"qty": qty}),
                              ("Delivery was requested to a residential address.", {})],
                "evaluate": evaluate, "question": {"type": "choice", "instructions": "What happens to this order?",
                                                   "criteria": {"within_limit": "Processed normally", "slightly_over": "Held for review", "far_over": "Cancelled"}}}
    return build(rng.randint(1, limit)), build(rng.choice([rng.randint(limit + 1, 2 * limit), rng.randint(2 * limit + 1, 4 * limit)]))


def family_deadline(rng):
    name = rng.choice(NAMES); due = date(2026, rng.randint(2, 11), rng.randint(1, 28)); grace = rng.choice([3, 7, 14])
    def evaluate(f):
        if not _need(f, "received", "due"): return UNDETERMINED
        late = (f["received"] - f["due"]).days
        return 0 if late <= 0 else 1 if late <= grace else 2
    def build(offset):
        received = due + timedelta(days=offset)
        return {"policy": f"Reports received by the deadline are on time. Reports received within {grace} days after the deadline are late but accepted. Later reports are refused.",
                "sentences": [(f"The filing deadline for {name}'s report was {_day(due)}.", {"due": due}),
                              (f"The report was received on {_day(received)}.", {"received": received}),
                              ("The report was submitted through the online portal.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "How late is this report?", "criteria": ["On time", "Late but accepted", "Refused"]}}
    return build(-rng.randint(0, 10)), build(rng.choice([rng.randint(1, grace), grace + rng.randint(1, 20)]))


def family_warranty_claim(rng):
    name = rng.choice(NAMES); item = rng.choice(ITEMS); bought = date(2026, rng.randint(1, 6), rng.randint(1, 28))
    standard, extended = rng.choice([(90, 365), (180, 730), (365, 1095)]); part = rng.choice(['stitching', 'battery', 'housing', 'zipper', 'switch'])
    def evaluate(f):
        if not _need(f, "claim", "purchase"): return UNDETERMINED
        age = (f["claim"] - f["purchase"]).days
        return 0 if age <= standard else 1 if age <= extended else 2
    def build(age):
        claim = bought + timedelta(days=age)
        return {"policy": f"Warranty claims made within {standard} days of purchase are covered in full. Claims made after that but within {extended} days are covered at half cost. Later claims are not covered.",
                "sentences": [(f"{name} purchased {item} on {_day(bought)}.", {"purchase": bought}),
                              (f"A warranty claim for it was filed on {_day(claim)}.", {"claim": claim}),
                              (f"The claim describes a defect in the {part}.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "How is this claim covered?", "criteria": ["Covered in full", "Covered at half cost", "Not covered"]}}
    a = rng.choice([rng.randint(1, standard), rng.randint(standard + 1, extended), extended + rng.randint(1, 200)])
    b = rng.choice([x for x in [rng.randint(1, standard), rng.randint(standard + 1, extended), extended + rng.randint(1, 200)] if evaluate({"claim": bought + timedelta(days=x), "purchase": bought}) != evaluate({"claim": bought + timedelta(days=a), "purchase": bought})] or [a])
    return build(a), build(b)


def family_sla_response(rng):
    name = rng.choice(NAMES); target, breach = rng.choice([(4, 24), (8, 48), (24, 72), (1, 8)])
    unit = "hours"; topic = rng.choice(['a login failure', 'a duplicate charge', 'a missing invoice', 'an export error']); queue = rng.choice(['email', 'chat', 'phone'])
    def evaluate(f):
        if not _need(f, "hours"): return UNDETERMINED
        return 0 if f["hours"] <= target else 1 if f["hours"] <= breach else 2
    def build(h):
        return {"policy": f"Support responses within {target} {unit} meet the service level. Responses after {target} but within {breach} {unit} are a minor breach. Anything slower is a major breach.",
                "sentences": [(f"{name} opened a priority ticket about {topic}.", {}),
                              (f"The first response arrived {h} {unit} after the ticket was opened.", {"hours": h}),
                              (f"The ticket was routed through the {queue} queue.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "How does this response time rate against the service level?", "criteria": ["Met", "Minor breach", "Major breach"]}}
    levels = [rng.randint(1, target), rng.randint(target + 1, breach), breach + rng.randint(1, 100)]
    a, b = rng.sample(levels, 2)
    return build(a), build(b)


def family_late_fee(rng):
    name = rng.choice(NAMES); grace, cap = rng.choice([(5, 30), (10, 60), (15, 45)]); amount = rng.choice([120, 450, 980, 2300]); method = rng.choice(['bank transfer', 'card', 'cheque'])
    def evaluate(f):
        if not _need(f, "days_late"): return UNDETERMINED
        return 0 if f["days_late"] <= grace else 1 if f["days_late"] <= cap else 2
    def build(d):
        return {"policy": f"Invoices paid within {grace} days after the due date incur no fee. Payments between {grace + 1} and {cap} days late incur a 2% fee. Payments later than {cap} days incur a 10% fee and a hold on the account.",
                "sentences": [(f"{name}'s invoice for ${amount:,} fell due last quarter.", {}),
                              (f"Payment was received {d} days after the due date.", {"days_late": d}),
                              (f"The payment was made by {method}.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "Which fee tier applies?", "criteria": ["No fee", "2% fee", "10% fee and account hold"]}}
    levels = [rng.randint(0, grace), rng.randint(grace + 1, cap), cap + rng.randint(1, 60)]
    a, b = rng.sample(levels, 2)
    return build(a), build(b)


def family_volume_discount(rng):
    name = rng.choice(NAMES); item = rng.choice(ITEMS); t1, t2 = rng.choice([(10, 50), (25, 100), (5, 20), (100, 500)]); dest = rng.choice(['warehouse', 'storefront', 'branch office'])
    def evaluate(f):
        if not _need(f, "units"): return UNDETERMINED
        return 0 if f["units"] < t1 else 1 if f["units"] < t2 else 2
    def build(u):
        return {"policy": f"Orders of fewer than {t1} units are charged the list price. Orders of {t1} to {t2 - 1} units receive the volume discount. Orders of {t2} units or more receive the wholesale rate.",
                "sentences": [(f"{name} placed a business order for {item}.", {}),
                              (f"The order is for {u} units.", {"units": u}),
                              (f"Delivery is to a {dest}.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "Which pricing tier applies?", "criteria": ["List price", "Volume discount", "Wholesale rate"]}}
    levels = [rng.randint(1, t1 - 1), rng.randint(t1, t2 - 1), t2 + rng.randint(0, t2)]
    a, b = rng.sample(levels, 2)
    return build(a), build(b)


def family_shipping_delay(rng):
    name = rng.choice(NAMES); item = rng.choice(ITEMS); promised = date(2026, rng.randint(1, 11), rng.randint(1, 28))
    minor, major = rng.choice([(1, 4), (2, 7), (3, 10)]); carrier = rng.choice(["the courier", "the postal service", "a freight partner"])
    def evaluate(f):
        if not _need(f, "delivered", "promised"): return UNDETERMINED
        late = (f["delivered"] - f["promised"]).days
        return 0 if late <= minor else 1 if late <= major else 2
    def build(offset):
        delivered = promised + timedelta(days=offset)
        return {"policy": f"Deliveries up to {minor} day{'s' if minor > 1 else ''} after the promised date count as on time. Deliveries {minor + 1} to {major} days after it are a minor delay and earn a shipping refund. Later deliveries are a major delay and earn a full refund.",
                "sentences": [(f"{name} ordered {item} with delivery promised for {_day(promised)}.", {"promised": promised}),
                              (f"The parcel was delivered on {_day(delivered)}.", {"delivered": delivered}),
                              (f"It was shipped by {carrier}.", {})],
                "evaluate": evaluate, "question": {"type": "score", "instructions": "How is this delivery classified?", "criteria": ["On time", "Minor delay: shipping refund", "Major delay: full refund"]}}
    levels = [rng.randint(-3, minor), rng.randint(minor + 1, major), major + rng.randint(1, 30)]
    a, b = rng.sample(levels, 2)
    return build(a), build(b)


FAMILIES = {"return_window": family_return_window, "spend_threshold": family_spend_threshold, "authorization": family_authorization,
            "age_eligibility": family_age_eligibility, "quantity_limit": family_quantity_limit, "deadline": family_deadline,
            "warranty_claim": family_warranty_claim, "sla_response": family_sla_response, "late_fee": family_late_fee,
            "volume_discount": family_volume_discount, "shipping_delay": family_shipping_delay}


def label_of(item, drop=None):
    facts = {}
    for i, (_, f) in enumerate(item["sentences"]):
        if i != drop: facts.update(f)
    return item["evaluate"](facts)


def check_pair(a, b):
    """None if the pair is valid (labels differ, exactly one sentence differs, evidence ablation and filler invariance
    hold), else the reason it is rejected."""
    la, lb = label_of(a), label_of(b)
    if UNDETERMINED in (la, lb): return "label_undetermined"
    if la == lb: return "labels_equal"
    if sum(x[0] != y[0] for x, y in zip(a["sentences"], b["sentences"])) != 1 or len(a["sentences"]) != len(b["sentences"]):
        return "not_exactly_one_sentence_differs"
    for item, label in ((a, la), (b, lb)):
        evidence = 0
        for i, (_, facts) in enumerate(item["sentences"]):
            got = label_of(item, drop=i)
            if facts and got != UNDETERMINED: return "ablation_failed"
            if not facts and got != label: return "invariance_failed"
            evidence += bool(facts)
        if evidence < 1: return "no_evidence_sentence"
    return None


def contrastive_generate(n_pairs_per_family, seed, families):
    records, report = [], {}
    for family in families:
        rng = random.Random(f"{seed}:{family}")
        kept, reasons, attempts = 0, Counter(), 0
        while kept < n_pairs_per_family and attempts < 50 * n_pairs_per_family:
            attempts += 1
            a, b = FAMILIES[family](rng)
            why = check_pair(a, b)
            if why:
                reasons[why] += 1
                continue
            pair_id = f"{seed}-{family}-{kept:04d}"
            order_seed = rng.getrandbits(64)
            for item, sibling in ((a, "a"), (b, "b")):
                order_rng = random.Random(order_seed)
                order = list(range(len(item["sentences"]))); order_rng.shuffle(order)
                sentences = [item["sentences"][i][0] for i in order]
                q = {**item["question"], "label": label_of(item), "src": f"contrastive_{family}"}
                records.append({"state": {"policy": item["policy"], "case": " ".join(sentences)}, "questions": {"decision": q},
                                "_meta": {"id": f"contrastive/{family}/{pair_id}/{sibling}", "group_id": f"contrastive/{family}/{pair_id}",
                                          "source": "contrastive", "family": family, "sibling": sibling}})
            kept += 1
        if kept < n_pairs_per_family:
            raise ValueError(f"{family}: only {kept}/{n_pairs_per_family} pairs passed checks ({dict(reasons)})")
        report[family] = {"pairs": kept, "rejected": dict(reasons)}
    return records, report


# --- hand-written customer-support showcase set --------------------------------------------------------------------------

SUPPORT_QUESTIONS = {
    "intent": {"type": "choice", "instructions": "Select the customer's main request.",
               "criteria": {"refund": "Return money for a charge or purchase", "cancel": "End a subscription or service",
                            "technical_issue": "Something in the product is broken or not working", "account_access": "Cannot log in or get into the account",
                            "shipping": "Where an order is or when it will arrive", "billing_question": "Explain or change a bill, plan or payment method"}},
    "mentions_charge": {"type": "noul", "instructions": "Does the message say the customer was charged money?"},
    "urgency": {"type": "score", "instructions": "Rate the urgency expressed in the message.", "criteria": ["routine", "urgent", "immediate"]},
}
SUPPORT = [
    ("I was charged twice for my March invoice and need the duplicate refunded.", {"intent": "refund", "mentions_charge": True, "urgency": 1}),
    ("Please cancel my subscription at the end of this billing cycle. No rush, just don't renew it.", {"intent": "cancel", "mentions_charge": False, "urgency": 0}),
    ("The app crashes every time I open the reports tab. It started after yesterday's update.", {"intent": "technical_issue", "mentions_charge": False, "urgency": 1}),
    ("I'm locked out of my account and our payroll run is due in an hour. I need access right now!", {"intent": "account_access", "mentions_charge": False, "urgency": 2}),
    ("My order was supposed to arrive last Tuesday and the tracking page hasn't updated in a week. Where is it?", {"intent": "shipping", "mentions_charge": False, "urgency": 1}),
    ("Could you explain the line called 'platform fee' on my latest bill? I don't remember agreeing to it.", {"intent": "billing_question", "mentions_charge": True, "urgency": 0}),
    ("Someone just charged $2,400 to my card through your store and it wasn't me. Refund it immediately and stop any further charges.", {"intent": "refund", "mentions_charge": True, "urgency": 2}),
    ("Whenever you get a chance, I'd like to switch my plan from monthly to annual billing.", {"intent": "billing_question", "mentions_charge": False, "urgency": 0}),
]


# --- record construction ------------------------------------------------------------------------------------------------

def to_prepared(labelled, *, record_id, group_id, source_family, schema_family, split, provenance):
    """Labelled request -> prepared record: the public request (no labels), gold keys, provenance."""
    request = {"state": labelled["state"], "questions": {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
                                                          for qid, q in labelled["questions"].items()}}
    req = DecisionRequest.model_validate(request)
    supervision = {}
    for qid, q in req.questions.items():
        y = labelled["questions"][qid]["label"]
        gold = y if q.type == "choice" else ("true" if y else "false") if q.type == "noul" else str(int(y))
        if gold not in question_keys(q.type, q.criteria):
            raise ValueError(f"{record_id}/{qid}: gold {gold!r} is not an option key")
        supervision[qid] = {"gold": gold, "task": labelled["questions"][qid].get("src", source_family)}
    return {"record_id": record_id, "group_id": group_id, "source_family": source_family, "schema_family": schema_family,
            "split": split, "request": request, "supervision": supervision, "provenance": provenance}


def context_size(tok, record, context):
    """Packed token count when the record fits the training context, else None."""
    rec, _ = to_record(DecisionRequest.model_validate(record["request"]))
    try:
        n = len(encode(tok, rec, max_state=context["max_state"], max_row=context["max_row"])["ids"])
    except AdmissionError:
        return None
    return n if n <= context["max_packed"] else None


def _load(repo, config, split, revision):
    from datasets import load_dataset
    return load_dataset(repo, config, split=split, revision=revision)


def _row_text(row, field):
    return row[field] if isinstance(row.get(field), str) else json.dumps(row, sort_keys=True)


def _public_records(name, spec, rows, split_name, revision, rng, names, variant=False):
    repo, config, _, _, _, fn, schema_family, field = spec
    out = []
    for i, row in rows:
        th = text_hash(_row_text(row, field))
        labelled = fn(row, rng, names, variant=variant)
        rid = f"{name}/{split_name}/{i}" + ("/variant" if variant else "")
        out.append(to_prepared(labelled, record_id=rid, group_id=f"{name}/{th[:20]}", source_family=name,
                               schema_family=f"{name}.variant" if variant else schema_family, split=None,
                               provenance={"repo": repo, "config": config, "revision": revision, "split": split_name, "row": i, "text_sha256": th,
                                           "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()}))
    return out


def _groups(records):
    g = defaultdict(list)
    for r in records:
        g[r["group_id"]].append(r)
    return g


def _take_groups(groups, order, cap):
    """Whole groups in `order` while the record count stays within cap."""
    out = []
    for gid in order:
        if len(out) + len(groups[gid]) > cap:
            continue
        out.extend(groups[gid])
    return out


def _hash_order(gids, seed):
    return sorted(gids, key=lambda g: hashlib.sha256(f"{seed}:order:{g}".encode()).hexdigest())


def prepare(cfg, out_dir=None):
    """Build every split and the manifest from the recipe. Deterministic in the recipe's data section and the pinned
    dataset revisions."""
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    d = cfg["data"]
    seed, context = d["seed"], d["context"]
    out_dir = Path(out_dir or d["out"])
    tok = AutoTokenizer.from_pretrained(cfg["parent"], revision=cfg["revision"])
    hub = HfApi()
    splits = {s: [] for s in SPLITS}
    report = {}
    sources_manifest = {}

    def admit(records, rep, split):
        """Keep the records that fit the training context; count drops (zeros included) and the longest packed length."""
        kept = []
        rep[f"dropped_context_{split}"] += 0
        for r in records:
            n = context_size(tok, r, context)
            if n is None:
                rep[f"dropped_context_{split}"] += 1
                continue
            rep["max_packed_tokens"] = max(rep["max_packed_tokens"], n)
            r["split"] = split
            kept.append(r)
        return kept

    for name, spec in {**PUBLIC, **UNSEEN}.items():
        repo, config, branch, train_split, eval_split, fn, schema_family, field = spec
        # the recipe's pinned revision when it declares one (reproducible rebuilds), else the Hub head of `branch`, recorded
        pinned = ((d.get("sources") or {}).get(name) or {}).get("revision")
        info = hub.dataset_info(repo, revision=pinned or branch)
        revision = info.sha
        card = info.card_data.to_dict() if info.card_data else {}
        rep = Counter()
        rng = random.Random(source_seed(seed, name))
        ev = _load(repo, config, eval_split, revision)
        names = ev.features["label"].names if name == "banking77" else None
        eval_idx = [i for i in rng.sample(range(len(ev)), len(ev)) if _label_ok(ev[i])]
        if name in UNSEEN:
            # unseen panel: official evaluation rows, never trained
            recs = admit(_public_records(name, spec, [(i, ev[i]) for i in eval_idx[:d["unseen_per_source"] * 2]], eval_split, revision, rng, names), rep, "unseen")
            splits["unseen"].extend(recs[:d["unseen_per_source"]])
            sources_manifest[name] = {"repo": repo, "config": config, "revision": revision, "pinned_ref": branch or "main", "license": card.get("license"),
                                      "role": "eval-only unseen-source panel", "rule": f"{d['unseen_per_source']} official {eval_split} rows (deterministic order), context-filtered"}
            report[name] = dict(rep)
            continue
        # official evaluation rows: showcase first, then locked test; every official text is excluded from the train pool
        eval_hashes = {text_hash(_row_text(r, field)) for r in ev}
        n_show, n_test = d["showcase_per_source"][name], d["official_test_per_source"]
        official = admit(_public_records(name, spec, [(i, ev[i]) for i in eval_idx[:2 * (n_show + n_test)]], eval_split, revision, rng, names), rep, "official")
        og = _groups(official)
        order = list(dict.fromkeys(r["group_id"] for r in official))
        show = _take_groups(og, order, n_show)
        show_ids = {r["group_id"] for r in show}
        test = _take_groups(og, [g for g in order if g not in show_ids], n_test)
        for r in show: r["split"] = "showcase"
        for r in test: r["split"] = "test"
        # showcase variants: the same rows under a changed schema, same group id
        vrng = random.Random(source_seed(seed, f"{name}:variant"))
        base = show[: d["variants_per_source"]]
        variants = _public_records(name, spec, [(r["provenance"]["row"], ev[r["provenance"]["row"]]) for r in base], eval_split, revision, vrng, names, variant=True)
        variants = admit(variants, rep, "showcase")
        splits["showcase"].extend(show + variants)
        splits["test"].extend(test)
        # train-split pool
        tr = _load(repo, config, train_split, revision)
        pool_idx = rng.sample(range(len(tr)), min(d["pool_per_source"], len(tr)))
        rows = []
        for i in pool_idx:
            row = tr[i]
            if not _label_ok(row):
                rep["dropped_unlabelled"] += 1
                continue
            if text_hash(_row_text(row, field)) in eval_hashes:
                rep["excluded_official_overlap"] += 1
                continue
            rows.append((i, row))
        pool = _public_records(name, spec, rows, train_split, revision, rng, names)
        pool = admit(pool, rep, "pool")
        pg = _groups(pool)
        by_bucket = defaultdict(list)
        for gid in dict.fromkeys(r["group_id"] for r in pool):
            by_bucket[bucket(gid, seed)].append(gid)
        for part in ("train", "dev", "calibration", "test"):
            gids = _hash_order(by_bucket[part], seed)
            recs = _take_groups(pg, gids, d["train_cap_per_source"]) if part == "train" else [r for g in gids for r in pg[g]]
            rep[f"pool_{part}_before_cap"] = sum(len(pg[g]) for g in gids)
            for r in recs:
                r["split"] = part
            splits[part].extend(recs)
        sources_manifest[name] = {
            "repo": repo, "config": config, "revision": revision, "pinned_ref": branch or "main", "license": card.get("license"),
            "role": "trainable", "train_split": train_split, "official_eval_split": eval_split,
            "rule": (f"official {eval_split}: deterministic order, first {n_show} originals (whole groups) -> showcase, next {n_test} -> test; "
                     f"{train_split}: deterministic pool of {d['pool_per_source']} rows, rows whose normalized {field} occurs anywhere in the official "
                     f"{eval_split} split excluded, context filter, groups = normalized {field} text, first 8 bytes of sha256(seed:split:group), big-endian, mod 10 -> "
                     f"train 0-6 / dev 7 / calibration 8 / newly reserved test 9, then train capped at {d['train_cap_per_source']} records (whole groups, hash order)"),
            "locked_panels": {"test": "official + newly reserved (provenance.split tells which)", "showcase": "official only"},
            "teacher": "ineligible (77 options exceed the 26 letter labels): gold-only supervision" if name == "banking77" else "eligible where labels resolve",
        }
        report[name] = dict(rep)
        print(f"{name}: {dict(rep)}", flush=True)

    # executable policy states
    p = d["policy"]
    rep = Counter()
    fams = [f for f in FAMILIES if f != p["heldout_contrastive_family"]]
    comp = composition_generate(p["composition_groups_per_shape"], f"{seed}-pool", TRAIN_SHAPES)
    con, con_report = contrastive_generate(p["contrastive_pairs_per_family"], f"{seed}-pool", fams)
    policy_pool = [_policy_record(r, "pool") for r in comp + con]
    policy_pool = admit(policy_pool, rep, "pool")
    pg = _groups(policy_pool)
    by_bucket = defaultdict(list)
    for gid in dict.fromkeys(r["group_id"] for r in policy_pool):
        by_bucket[bucket(gid, seed)].append(gid)
    for part in ("train", "dev", "calibration", "test"):
        gids = _hash_order(by_bucket[part], seed)
        recs = _take_groups(pg, gids, p["train_cap"]) if part == "train" else [r for g in gids for r in pg[g]]
        rep[f"pool_{part}_before_cap"] = sum(len(pg[g]) for g in gids)
        for r in recs: r["split"] = part
        splits[part].extend(recs)
    held_dev = [_policy_record(r, "dev") for r in composition_generate(p["heldout_groups_per_shape"], f"{seed}-heldout-dev", DEV_SHAPES)]
    held_test = [_policy_record(r, "test") for r in composition_generate(p["heldout_groups_per_shape"], f"{seed}-heldout-test", TEST_SHAPES, styles=(0, 1, 2))]
    held_con, held_con_report = contrastive_generate(p["heldout_contrastive_pairs"], f"{seed}-heldout-test", [p["heldout_contrastive_family"]])
    held_test += [_policy_record(r, "test") for r in held_con]
    show_comp = composition_generate(p["showcase_groups_per_test_shape"], f"{seed}-showcase", TEST_SHAPES, styles=(0, 1, 2))
    show_con, _ = contrastive_generate(p["showcase_heldout_pairs"], f"{seed}-showcase", [p["heldout_contrastive_family"]])
    show_policy = [_policy_record(r, "showcase") for r in show_comp + show_con]
    splits["dev"].extend(admit(held_dev, rep, "dev"))
    splits["test"].extend(admit(held_test, rep, "test"))
    splits["showcase"].extend(admit(show_policy, rep, "showcase"))
    report["policy"] = dict(rep)
    print(f"policy: {dict(rep)}", flush=True)

    # hand-written customer-support showcase set
    support = []
    for i, (message, gold) in enumerate(SUPPORT):
        labelled = {"state": {"message": message}, "questions": {qid: {**copy.deepcopy(q), "label": gold[qid], "src": f"support_{qid}"} for qid, q in SUPPORT_QUESTIONS.items()}}
        support.append(to_prepared(labelled, record_id=f"support/{i}", group_id=f"support/{i}", source_family="support", schema_family="support.triage",
                                   split="showcase", provenance={"generator": "hand-written for the demo (decision_model.data.SUPPORT)", "text_sha256": text_hash(message)}))
    splits["showcase"].extend(admit(support, Counter(), "showcase"))

    # user-supplied labelled corpora (data.custom): split like a public pool; the test share joins the locked test panel
    custom_manifest = {}
    if d.get("custom"):
        forbidden = {t for split in ("showcase", "test") for r in splits[split] for t in (r["provenance"]["text_sha256"], *state_hashes(r["request"]["state"]))}
        for demo in sorted(Path(d.get("demo_dir", "demo")).glob("*.json")):
            forbidden |= state_hashes(json.loads(demo.read_text(encoding=ENCODING))["state"])
        for entry in d["custom"]:
            recs, info = custom_records(entry, seed, forbidden)
            for split in ("train", "dev", "calibration", "test"):
                splits[split].extend(admit([r for r in recs if r["split"] == split], info["report"], split))
            info["counts"] = {s: sum(r["source_family"] == entry["source_family"] for r in splits[s]) for s in ("train", "dev", "calibration", "test")}
            custom_manifest[entry["source_family"]] = info
            print(f"{entry['source_family']}: {info['counts']} {dict(info['report'])}", flush=True)

    _check_disjoint(splits)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1, "seed": seed, "parent": {"name": cfg["parent"], "revision": cfg["revision"]},
        "config_sha256": hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(),
        "context": context,
        "sources": sources_manifest,
        "policy": {
            "generators": "decision_model.data composition and contrastive generators",
            "train_shapes": list(TRAIN_SHAPES), "dev_heldout_shapes": list(DEV_SHAPES), "test_heldout_shapes": list(TEST_SHAPES),
            "trainable_contrastive_families": fams, "heldout_contrastive_family": p["heldout_contrastive_family"],
            "rule": (f"pool: {p['composition_groups_per_shape']} composition groups (4 states) per train shape + {p['contrastive_pairs_per_family']} "
                     f"contrastive pairs per trainable family; groups split 70/10/10/10 by the first 8 bytes of sha256(seed:split:group), big-endian, mod 10; train capped at {p['train_cap']} states. "
                     f"Held-out structures: DEV_SHAPES only in dev, TEST_SHAPES (render styles 0-2; style 2 never trains) and the "
                     f"{p['heldout_contrastive_family']} family only in test and showcase."),
            "contrastive_report": con_report, "heldout_contrastive_report": held_con_report,
        },
        "roles": {"train": "gradient updates, teacher targets, augmentation", "dev": "checkpoint selection (development NLL); never temperature",
                  "calibration": "temperature fit on the selected checkpoint", "test": "locked final evaluation (official + newly reserved + held-out policy structures)",
                  "showcase": "compact benchmark/demo panel: official rows, variants (<source>.variant, same group id), held-out policy, hand-written support set",
                  "unseen": "eval-only panel from sources absent from training (PAWS, QNLI, Emotion)"},
        "selection": "Normalized exact-text grouping and overlap exclusion; no fuzzy decontamination; no claim about parent pretraining contamination.",
        "custom": custom_manifest,
        "report": report, "files": {}, "counts": {},
    }
    for split, records in splits.items():
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, records)
        by_source, by_primitive, by_schema = Counter(), Counter(), Counter()
        for r in records:
            by_source[r["source_family"]] += 1
            by_schema[r["schema_family"]] += 1
            for q in r["request"]["questions"].values():
                by_primitive[q["type"]] += 1
        manifest["files"][path.name] = {"sha256": digest(path), "records": len(records), "questions": sum(by_primitive.values())}
        manifest["counts"][split] = {"by_source": dict(sorted(by_source.items())), "questions_by_primitive": dict(sorted(by_primitive.items())),
                                     "by_schema_family": dict(sorted(by_schema.items()))}
    manifest["code_sha256"] = {name: digest(Path(__file__).parent / name) for name in ("data.py", "render.py", "schema.py")}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def _policy_record(r, split):
    m = r["_meta"]
    source = m["source"]
    schema = f"{source}.{m['family']}"
    prov = {"generator": f"decision_model.data.{source}", "family": m["family"], "text_sha256": text_hash(json.dumps(r["state"], sort_keys=True))}
    if "certificate" in m:
        prov.update(pair_kind=m["pair_kind"], sibling=m["sibling"], render_style=m["render_style"], certificate=m["certificate"])
    else:
        prov.update(sibling=m["sibling"])
    return to_prepared(r, record_id=m["id"], group_id=m["group_id"], source_family=source, schema_family=schema, split=split, provenance=prov)


def state_hashes(state):
    """Normalized-text hashes that identify a state: its rendered text and every string field of at least five words
    (short fields such as a channel name or a customer name are shared by unrelated states and identify nothing)."""
    out = {text_hash(render_content(state))}
    stack = [state]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            if len(v.split()) >= 5:
                out.add(text_hash(v))
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
    return out


def custom_records(entry, seed, forbidden):
    """One data.custom file (labelled-request JSONL: state, questions{qid: {type, instructions, criteria, label}},
    schema_family, notes; labels are option ids, "true"/"false", or the level index as a string) -> prepared records
    with split assigned by the public-pool hash rule. Records whose state text collides with `forbidden` are excluded."""
    if entry.get("split", {"train": 0.7, "dev": 0.1, "calibration": 0.1, "test": 0.1}) != {"train": 0.7, "dev": 0.1, "calibration": 0.1, "test": 0.1}:
        raise ValueError("data.custom split must be the 70/10/10/10 group rule")
    path, fam = Path(entry["path"]), entry["source_family"]
    rep = Counter()
    out = []
    for n, line in enumerate(path.read_text(encoding=ENCODING).splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        hashes = state_hashes(raw["state"])
        if hashes & forbidden:
            rep["excluded_text_collision"] += 1
            continue
        qs = {}
        for qid, q in raw["questions"].items():
            y = q["label"]
            if q["type"] == "noul":
                y = y if isinstance(y, bool) else {"true": True, "false": False}[str(y).lower()]
            elif q["type"] == "score":
                y = int(y)
            qs[qid] = {**q, "label": y, "src": f"{fam}_{qid}"}
        th = text_hash(render_content(raw["state"]))
        gid = f"{fam}/{th[:20]}"
        rec = to_prepared({"state": raw["state"], "questions": qs}, record_id=f"{fam}/{n}", group_id=gid, source_family=fam,
                          schema_family=raw.get("schema_family", f"{fam}.custom"), split=bucket(gid, seed),
                          provenance={"path": str(path), "line": n, "text_sha256": th, "notes": raw.get("notes")})
        out.append(rec)
    rep["read"] = len(out) + rep["excluded_text_collision"]
    return out, {"path": str(path), "sha256": digest(path), "report": rep,
                 "rule": ("groups = normalized rendered-state text; first 8 bytes of sha256(seed:split:group), big-endian, mod 10 -> "
                          "train 0-6 / dev 7 / calibration 8 / test 9 (joins the locked test panel); records whose rendered state or any "
                          "string field of 5+ words matches a showcase, test or demo/*.json text are excluded; context filter; no cap")}


def _check_disjoint(splits):
    """No group and no normalized text in two splits."""
    owner_g, owner_t = {}, {}
    for split, records in splits.items():
        for r in records:
            for owner, key in ((owner_g, r["group_id"]), (owner_t, (r["source_family"], r["provenance"]["text_sha256"]))):
                if owner.setdefault(key, split) != split:
                    raise ValueError(f"{key} appears in both {owner[key]} and {split}")


def build_panel(cfg, out_path="data/panels/extended.jsonl", per_source=100, prepared=None):
    """Extended never-trained evaluation panel (standalone; data/prepared*/manifest.json untouched). Official test rows in
    a deterministic order (recipe seed), converted with Kev's converters, context-filtered, and excluded when any state
    text matches a train/dev/calibration text of the prepared corpus."""
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    d = cfg["data"]
    seed, context = d["seed"], d["context"]
    prepared = Path(prepared or d["out"])
    tok = AutoTokenizer.from_pretrained(cfg["parent"], revision=cfg["revision"])
    trained = {h for s in ("train", "dev", "calibration") for r in load_split(prepared, s) for h in state_hashes(r["request"]["state"])}
    hub = HfApi()
    records, sources = [], {}
    for name, spec in EXTENDED.items():
        repo, config, branch, _, split, fn, schema_family, field = spec
        info = hub.dataset_info(repo, revision=branch)
        revision = info.sha
        rep = Counter()
        rng = random.Random(source_seed(seed, f"extended:{name}"))
        ds = _load(repo, config, split, revision)
        names = ds.features["label"].names if name == "dbpedia14" else None
        order = [i for i in rng.sample(range(len(ds)), len(ds))]
        kept = []
        for i in order:
            if len(kept) >= per_source:
                break
            row = ds[i]
            if not _label_ok(row) or (name == "arc" and row["answerKey"] not in row["choices"]["label"]):
                rep["dropped_unlabelled"] += 1
                continue
            r = _public_records(name, spec, [(i, row)], split, revision, rng, names)[0]
            if state_hashes(r["request"]["state"]) & trained:
                rep["excluded_text_collision"] += 1
                continue
            if context_size(tok, r, context) is None:
                rep["dropped_context"] += 1
                continue
            DecisionRequest.model_validate(r["request"])
            r["split"] = "extended"
            kept.append(r)
        prim = Counter(q["type"] for r in kept for q in r["request"]["questions"].values())
        sources[name] = {"repo": repo, "config": config, "split": split, "revision": revision, "pinned_ref": branch or "main",
                         "license": (info.card_data.to_dict() if info.card_data else {}).get("license"), "records": len(kept),
                         "questions_by_primitive": dict(prim), "considered": len(kept) + sum(rep.values()),
                         "exclusions": {k: rep[k] for k in ("excluded_text_collision", "dropped_context", "dropped_unlabelled")}}
        records.extend(kept)
        print(f"extended {name}: {len(kept)} records {dict(rep)}", flush=True)
    _check_disjoint({"extended": records})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, records)
    manifest = {"version": 1, "role": "extended never-trained evaluation panel; not part of any conversion's corpus", "seed": seed,
                "tokenizer": {"parent": cfg["parent"], "revision": cfg["revision"], "note": "Qwen3 sizes share this tokenizer"},
                "context": context, "per_source": per_source,
                "rule": (f"official {', '.join(sorted({v[4] for v in EXTENDED.values()}))} rows in a deterministic order (sha256(seed:extended:<source>) seeded), "
                         f"first {per_source} kept after exclusions"),
                "exclusion_check": {"against": str(prepared), "splits": ["train", "dev", "calibration"],
                                    "prepared_manifest_sha256": digest(prepared / "manifest.json"),
                                    "method": "normalized hash of the rendered state and of every string field of 5+ words (data.state_hashes)"},
                "sources": sources, "records": len(records),
                "questions_by_primitive": dict(Counter(q["type"] for r in records for q in r["request"]["questions"].values())),
                "file": {"path": str(out), "sha256": digest(out)}, "code_sha256": digest(Path(__file__))}
    write_json(out.with_name(out.stem + ".manifest.json"), manifest)
    return manifest


def load_recipe(path):
    return yaml.safe_load(Path(path).read_text(encoding=ENCODING))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="recipes/qwen3-1.7b.yaml")
    ap.add_argument("--out", default=None, help="output directory (default: the recipe's data.out)")
    ap.add_argument("--panel", choices=["extended"], help="build the extended evaluation panel instead of the corpus")
    a = ap.parse_args()
    if a.panel:
        m = build_panel(load_recipe(a.config), a.out or "data/panels/extended.jsonl")
        print(json.dumps({k: m[k] for k in ("records", "questions_by_primitive", "file")}, indent=2))
        return
    m = prepare(load_recipe(a.config), a.out)
    print(json.dumps(m["files"], indent=2))


if __name__ == "__main__":
    main()
