"""The prepared corpus every later stage consumes (trainer, teacher collector, calibration, locked evaluation) keeps its
split discipline: a leak here silently inflates every reported number."""
import json
from pathlib import Path

from decision_model.data import SPLITS, digest, read_jsonl, state_hashes
from decision_model.schema import DecisionRequest, question_keys

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "prepared"


def test_prepared_splits_are_disjoint_valid_and_match_the_manifest():
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    custom = set(manifest.get("custom", {}))
    held = {h for p in (ROOT / "demo").glob("*.json") for h in state_hashes(json.loads(p.read_text())["state"])}
    trained = []
    owner, official = {}, {s["repo"]: s.get("official_eval_split") for s in manifest["sources"].values()}
    for split in SPLITS:
        path = PREPARED / f"{split}.jsonl"
        records = read_jsonl(path)
        assert manifest["files"][path.name] == {"sha256": digest(path), "records": len(records),
                                                "questions": sum(len(r["request"]["questions"]) for r in records)}
        for r in records:
            assert r["split"] == split
            # a group lives in exactly one split
            assert owner.setdefault(r["group_id"], split) == split, f"{r['group_id']} in {owner[r['group_id']]} and {split}"
            # official held-out rows never reach train/dev/calibration
            prov = r["provenance"]
            if split in ("train", "dev", "calibration"):
                assert prov.get("split") != official.get(prov.get("repo"), "<none>"), r["record_id"]
            if split in ("showcase", "test"):
                held |= state_hashes(r["request"]["state"])
            elif r["source_family"] in custom and split != "unseen":
                trained.append(r)
            # the request is a valid public request and every gold label is one of its option keys
            req = DecisionRequest.model_validate(r["request"])
            assert set(r["supervision"]) == set(req.questions)
            for qid, q in req.questions.items():
                assert r["supervision"][qid]["gold"] in question_keys(q.type, q.criteria)
    # a user-supplied corpus never trains on a showcase, locked-test or demo text
    assert not [r["record_id"] for r in trained if state_hashes(r["request"]["state"]) & held]
