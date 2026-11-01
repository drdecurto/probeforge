from pathlib import Path
from .io import canonical, read_jsonl, file_hash
from .environment import public_state, consistent
from .tasks import TRAIN_FAMILIES, OOD_FAMILIES
import json


def audit_dataset(path):
    root = Path(path)
    manifest = json.loads((root/"manifest.json").read_text())
    seen, total = set(), 0
    for split in ("train", "validation", "test_id", "test_ood"):
        if file_hash(root/f"{split}.jsonl") != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Checksum mismatch: {split}")
        rows = list(read_jsonl(root/f"{split}.jsonl"))
        groups = {}
        for task in rows:
            if task["split"] != split:
                raise ValueError("Split mismatch")
            allowed = OOD_FAMILIES if split == "test_ood" else TRAIN_FAMILIES
            if task["family"] not in allowed:
                raise ValueError("Structural holdout violation")
            groups.setdefault(task["group_id"], []).append(task)
            if task["truth_index"] not in consistent(public_state(task)):
                raise ValueError("Initial observation contradicts truth")
        if seen & groups.keys():
            raise ValueError("Counterfactual group split leakage")
        seen.update(groups)
        for group in groups.values():
            if len({canonical(public_state(t)) for t in group}) != 1:
                raise ValueError("Public information differs between paired worlds")
            if sorted(t["truth_index"] for t in group) != list(range(len(group[0]["candidates"]))):
                raise ValueError("Incomplete counterfactual group")
        total += len(rows)
    return {"passed": True, "groups": len(seen), "worlds": total}
