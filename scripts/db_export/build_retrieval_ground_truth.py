import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET_DIR = ROOT / "data" / "epo_txt"
PAIR_ROOT = ROOT / "data"
OUT_DIR = ROOT / "data" / "processed"

TARGET_PATTERN = "epo_txt_EP*.json"
PAIR_PATTERN = "**/epo_srpt_label_*.json"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Retrieval corpus patent IDs
# ============================================================

target_ids = {
    str(load_json(p).get("patent_id", "")).strip()
    for p in TARGET_DIR.glob(TARGET_PATTERN)
}

target_ids.discard("")

print("Target patents:", len(target_ids))


# ============================================================
# Output
# ============================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

labeled_path = OUT_DIR / "sr_pair_labeled.jsonl"
gt_path = OUT_DIR / "retrieval_ground_truth.jsonl"

ground_truth = defaultdict(
    lambda: defaultdict(set)
)

pair_files = sorted(
    PAIR_ROOT.glob(PAIR_PATTERN)
)

print("Pair files:", len(pair_files))


# ============================================================
# 1. sr_pair_labeled.jsonl
# ============================================================

with open(labeled_path, "w", encoding="utf-8") as out:

    for path in pair_files:

        data = load_json(path)
        records = data if isinstance(data, list) else [data]

        for r in records:

            query_id = str(
                r.get("querypatent_id", "")
            ).strip()

            cited_id = str(
                r.get("cited_patent_id", "")
            ).strip()

            if not query_id or not cited_id:
                continue

            label = str(
                r.get("epo_label", "")
            ).strip()

            categories = [
                x.strip().upper()
                for x in label.split(",")
                if x.strip()
            ]

            usable = (
                query_id in target_ids
                and cited_id in target_ids
            )

            row = {
                **r,
                "categories": categories,
                "conflict_label": int("X" in categories),
                "is_x": "X" in categories,
                "is_y": "Y" in categories,
                "is_a": "A" in categories,
                "query_in_retrieval_corpus":
                    query_id in target_ids,
                "cited_in_retrieval_corpus":
                    cited_id in target_ids,
                "usable_for_retrieval": usable,
            }

            out.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

            if usable:
                ground_truth[
                    query_id
                ][
                    cited_id
                ].update(categories)


# ============================================================
# 2. retrieval_ground_truth.jsonl
# ============================================================

with open(gt_path, "w", encoding="utf-8") as out:

    for query_id, cited_map in sorted(
        ground_truth.items()
    ):

        all_cited = sorted(
            cited_map.keys()
        )

        conflict = sorted(
            pid
            for pid, cats in cited_map.items()
            if "X" in cats
        )

        background = sorted(
            pid
            for pid, cats in cited_map.items()
            if "X" not in cats
        )

        row = {
            "query_patent_id": query_id,
            "all_cited_patents": all_cited,
            "conflict_patents": conflict,
            "background_patents": background,
            "relevance": {
                pid: {
                    "categories": sorted(cats),
                    "conflict_label": int("X" in cats),
                }
                for pid, cats in cited_map.items()
            },
            "num_all_cited": len(all_cited),
            "num_conflict": len(conflict),
            "num_background": len(background),
        }

        out.write(
            json.dumps(
                row,
                ensure_ascii=False
            ) + "\n"
        )


print("Saved:", labeled_path)
print("Saved:", gt_path)
