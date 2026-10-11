import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parents[2]
QDRANT_PATH = ROOT / "data" / "qdrant_text_units"
GT_PATH = ROOT / "data/processed/retrieval_ground_truth.jsonl"
OUT_PATH = ROOT / "data/processed/text_retrieval_results.csv"

COLLECTIONS = {
    "Qwen3": "patent_text_units_qwen3_0_6b",
    "Arctic": "patent_text_units_arctic_l_v2",
    "NV-Embed": "patent_text_units_nv_embed_v2",
}

UNITS = ["document", "whole_claims", "claim"]
KS = [1, 5, 10, 50, 100]
MAX_K = max(KS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_gt():
    with open(GT_PATH, encoding="utf-8") as f:
        return {
            r["query_patent_id"]: r
            for r in map(json.loads, f)
        }


def load_vectors(client, collection, unit):
    filt = models.Filter(
        must=[
            models.FieldCondition(
                key="unit_type",
                match=models.MatchValue(value=unit),
            )
        ]
    )

    points, offset = [], None

    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=2000,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)

        if offset is None:
            break

    patent_ids = [p.payload["patent_id"] for p in points]

    vectors = np.asarray(
        [p.vector for p in points],
        dtype=np.float32,
    )

    vectors = F.normalize(
        torch.from_numpy(vectors).to(DEVICE),
        dim=1,
    )

    return patent_ids, vectors


def make_rankings(patent_ids, vectors, query_ids):
    patents = sorted(set(patent_ids))
    patent_to_idx = {p: i for i, p in enumerate(patents)}

    unit_by_patent = defaultdict(list)

    for i, patent_id in enumerate(patent_ids):
        unit_by_patent[patent_id].append(i)

    unit_patent_idx = torch.tensor(
        [patent_to_idx[p] for p in patent_ids],
        device=DEVICE,
    )

    rankings = {}

    for n, query_id in enumerate(query_ids, 1):
        q_idx = unit_by_patent.get(query_id)

        if not q_idx:
            continue

        q = vectors[q_idx]

        # max over all query units
        unit_scores = (q @ vectors.T).amax(dim=0)

        # max over all candidate units belonging to each patent
        patent_scores = torch.full(
            (len(patents),),
            -torch.inf,
            device=DEVICE,
        )

        patent_scores.scatter_reduce_(
            0,
            unit_patent_idx,
            unit_scores,
            reduce="amax",
            include_self=True,
        )

        # exclude query patent itself
        patent_scores[patent_to_idx[query_id]] = -torch.inf

        top_idx = torch.topk(
            patent_scores,
            min(MAX_K, len(patents) - 1),
        ).indices.tolist()

        rankings[query_id] = [patents[i] for i in top_idx]

        if n % 100 == 0:
            print(f"  {n}/{len(query_ids)} queries")

    return rankings


def evaluate(rankings, gt, key):
    rows = []

    for query_id, truth in gt.items():
        relevant = set(truth[key])

        if not relevant or query_id not in rankings:
            continue

        ranking = rankings[query_id]
        hits = [int(p in relevant) for p in ranking]

        result = {
            f"R@{k}": sum(hits[:k]) / len(relevant)
            for k in KS
        }

        result["MRR@100"] = next(
            (1 / (i + 1) for i, h in enumerate(hits) if h),
            0,
        )

        result["MAP@100"] = (
            sum(
                sum(hits[:i + 1]) / (i + 1)
                for i, h in enumerate(hits)
                if h
            )
            / len(relevant)
        )

        k = 10
        dcg = sum(
            h / math.log2(i + 2)
            for i, h in enumerate(hits[:k])
        )

        idcg = sum(
            1 / math.log2(i + 2)
            for i in range(min(len(relevant), k))
        )

        result["nDCG@10"] = dcg / idcg if idcg else 0
        rows.append(result)

    return {
        "Queries": len(rows),
        **{
            metric: np.mean([r[metric] for r in rows])
            for metric in rows[0]
        },
    }


gt = load_gt()
client = QdrantClient(path=str(QDRANT_PATH))
results = []

print("Device:", DEVICE)
print("Queries:", len(gt))

for model_name, collection in COLLECTIONS.items():
    for unit in UNITS:
        print(f"\n{model_name} | {unit}")

        patent_ids, vectors = load_vectors(
            client,
            collection,
            unit,
        )

        print("Vectors:", len(patent_ids))

        rankings = make_rankings(
            patent_ids,
            vectors,
            gt.keys(),
        )

        for task, key in [
            ("Citation", "all_cited_patents"),
            ("Conflict-X", "conflict_patents"),
        ]:
            scores = evaluate(rankings, gt, key)

            row = {
                "Model": model_name,
                "Granularity": unit,
                "Task": task,
                **scores,
            }

            results.append(row)

            print(
                task,
                "|",
                "R@10:",
                f"{scores['R@10']:.4f}",
                "|",
                "MRR:",
                f"{scores['MRR@100']:.4f}",
            )

        del vectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys(),
    )
    writer.writeheader()

    for row in results:
        writer.writerow({
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in row.items()
        })

print("\nSaved:", OUT_PATH)
