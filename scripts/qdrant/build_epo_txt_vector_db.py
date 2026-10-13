import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from sentence_transformers.models import (
    Transformer,
    Pooling,
    Normalize,
)

"""
Patent P1
├─ document vector
├─ whole_claims vector
├─ claim 1 vector
├─ claim 2 vector
└─ ...

+ payload
  ├ patent_id
  ├ family_id
  ├ unit_type
  ├ unit_id
  ├ claim_no
  ├ text
  ├ IPC/CPC
  └ etc metadata


Document Vector
2,376

Whole Claims Vector
2,376

Individual Claim Vector
33,217
────────────────────────
Total
37,969 vectors

"""
# ============================================================
# Default settings.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "epo_txt"

QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_text_units"

# ============================================================
# Embedding model settings.
# ============================================================

MODEL_CONFIGS = {

    "qwen": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "collection_name": "patent_text_units_qwen3_0_6b",
        "max_seq_length": 32768,
        "trust_remote_code": False,
        "add_eos": False,
        "default_batch_size": 2,
    },

    "nvembed": {
        "model_name": "nvidia/NV-Embed-v2",
        "collection_name": "patent_text_units_nv_embed_v2",
        "max_seq_length": 32768,
        "trust_remote_code": True,
        "add_eos": True,
        "default_batch_size": 1,
    },

    "arctic": {
        "model_name": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "collection_name": "patent_text_units_arctic_l_v2",
        "max_seq_length": 8192,
        "trust_remote_code": False,
        "add_eos": False,
        "default_batch_size": 8,
    },
}

# Current dataset-analysis results:
# document max      = 1,369
# claim max         = 17,260
# whole_claims max  = 29,990


# ============================================================
# Command-line arguments.
# ============================================================

def parse_args():


    parser = argparse.ArgumentParser(
        description=(
            "Build Qdrant Local vector DB for "
            "document / whole claims / individual claims."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_CONFIGS.keys(),
        required=True,
        help="Embedding model: qwen / nvembed / arctic",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="epo_txt_EP*.json",
    )

    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=QDRANT_PATH,
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Optional custom collection name.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size. Uses model default if omitted.",
    )

    parser.add_argument(
        "--upsert-size",
        type=int,
        default=128,
        help="Number of text units sent to Qdrant at once.",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate collection.",
    )

    return parser.parse_args()


# ============================================================
# GPU
# ============================================================

def get_device():

    if torch.cuda.is_available():

        print(
            f"[GPU] {torch.cuda.get_device_name(0)}"
        )

        print(
            f"[VRAM] "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

        return "cuda"

    print("[WARNING] CUDA unavailable. Using CPU.")

    return "cpu"

# ============================================================
# Embedding model loader
# ============================================================

# ============================================================
# Embedding model loader
# ============================================================

def load_embedding_model(config, device):

    print()
    print(f"Loading model: {config['model_name']}")

    # ========================================================
    # NV-Embed-v2
    #
    # sentence-transformers 2.7.0 does not support
    # SentenceTransformer(model_kwargs=...); instantiate the
    # Transformer module directly for FP16 loading.
    # ========================================================

    if config["model_name"] == "nvidia/NV-Embed-v2":

        transformer = Transformer(
            config["model_name"],

            max_seq_length=
                config["max_seq_length"],

            model_args={
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
            },

            tokenizer_args={
                "trust_remote_code": True,
                "padding_side": "right",
            },
        )

        # Official NV-Embed-v2 pooling configuration.
        pooling = Pooling(
            transformer.get_word_embedding_dimension(),

            pooling_mode_cls_token=False,
            pooling_mode_mean_tokens=True,
            pooling_mode_max_tokens=False,
            pooling_mode_mean_sqrt_len_tokens=False,
            pooling_mode_weightedmean_tokens=False,
            pooling_mode_lasttoken=False,

            include_prompt=False,
        )

        normalize = Normalize()

        model = SentenceTransformer(
            modules=[
                transformer,
                pooling,
                normalize,
            ],
            device=device,
        )

    # ========================================================
    # Qwen / Arctic
    # ========================================================

    else:

        model = SentenceTransformer(
            config["model_name"],
            device=device,
            trust_remote_code=
                config["trust_remote_code"],
        )


    # ========================================================
    # Shared settings.
    # ========================================================

    model.max_seq_length = (
        config["max_seq_length"]
    )

    if config.get(
        "add_eos",
        False
    ):
        model.tokenizer.padding_side = "right"


    print(
        f"Embedding dimension: "
        f"{model.get_sentence_embedding_dimension()}"
    )

    print(
        f"Model max seq length: "
        f"{model.max_seq_length}"
    )

    print(
        f"Model dtype: "
        f"{next(model.parameters()).dtype}"
    )

    return model

# ============================================================
# Deterministic Qdrant ID
# ============================================================

def make_point_id(
    patent_id: str,
    unit_type: str,
    unit_id: str,
):

    key = (
        f"{patent_id}:"
        f"{unit_type}:"
        f"{unit_id}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            key
        )
    )


# ============================================================
# SR_Target → text units
# ============================================================

def iter_text_units(files):

    for file_path in files:

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                patent = json.load(f)

        except Exception as e:

            print(
                f"[ERROR] {file_path.name}: {e}"
            )

            continue

        # ----------------------------------------------------
        # Patent metadata
        # ----------------------------------------------------

        patent_id = str(
            patent.get(
                "patent_id",
                ""
            )
        ).strip()

        if not patent_id:

            print(
                f"[WARNING] patent_id missing: "
                f"{file_path.name}"
            )

            continue

        record_id = patent.get(
            "record_id",
            ""
        )

        family_id = patent.get(
            "family_id",
            ""
        )

        pub_date = patent.get(
            "pub_date",
            ""
        )

        app_date = patent.get(
            "app_date",
            ""
        )

        title = str(
            patent.get(
                "title",
                ""
            )
        ).strip()

        abstract = str(
            patent.get(
                "abstract",
                ""
            )
        ).strip()

        ipc_codes = patent.get(
            "ipc_codes",
            []
        )

        cpc_codes = patent.get(
            "cpc_codes",
            []
        )

        claims = patent.get(
            "claims",
            []
        )

        # ====================================================
        # 1. DOCUMENT VECTOR
        #
        # Title + Abstract
        # ====================================================

        document_parts = []

        if title:

            document_parts.append(
                f"Title: {title}"
            )

        if abstract:

            document_parts.append(
                f"Abstract: {abstract}"
            )

        document_text = "\n".join(
            document_parts
        ).strip()

        if document_text:

            yield {
                "record_id": record_id,
                "patent_id": patent_id,
                "family_id": family_id,
                "pub_date": pub_date,
                "app_date": app_date,
                "title": title,
                "ipc_codes": ipc_codes,
                "cpc_codes": cpc_codes,

                "unit_type": "document",
                "unit_id": "document",

                "text": document_text,

                "num_claims": len(claims),

                "source_file": file_path.name,
            }


        # ====================================================
        # Normalize claim text.
        # ====================================================

        valid_claims = []

        for claim in claims:

            claim_no = claim.get(
                "claim_no"
            )

            claim_text = str(
                claim.get(
                    "text",
                    ""
                )
            ).strip()

            if claim_no is None:
                continue

            if not claim_text:
                continue

            valid_claims.append(
                {
                    "claim_no": int(
                        claim_no
                    ),
                    "text": claim_text,
                }
            )


        # ====================================================
        # 2. WHOLE CLAIMS VECTOR
        # ====================================================

        if valid_claims:

            whole_claims_text = "\n".join(

                f"Claim {claim['claim_no']}: "
                f"{claim['text']}"

                for claim in valid_claims
            )

            yield {
                "record_id": record_id,
                "patent_id": patent_id,
                "family_id": family_id,
                "pub_date": pub_date,
                "app_date": app_date,
                "title": title,
                "ipc_codes": ipc_codes,
                "cpc_codes": cpc_codes,

                "unit_type": "whole_claims",
                "unit_id": "whole_claims",

                "text": whole_claims_text,

                "num_claims": len(
                    valid_claims
                ),

                "source_file": file_path.name,
            }


        # ====================================================
        # 3. INDIVIDUAL CLAIM VECTORS
        # ====================================================

        for claim in valid_claims:

            claim_no = claim[
                "claim_no"
            ]

            claim_text = claim[
                "text"
            ]

            yield {
                "record_id": record_id,
                "patent_id": patent_id,
                "family_id": family_id,
                "pub_date": pub_date,
                "app_date": app_date,
                "title": title,
                "ipc_codes": ipc_codes,
                "cpc_codes": cpc_codes,

                "unit_type": "claim",
                "unit_id": str(
                    claim_no
                ),

                "claim_no": claim_no,

                "text": claim_text,

                "num_claims": len(
                    valid_claims
                ),

                "source_file": file_path.name,
            }


# ============================================================
# Batch Embedding + Qdrant insert
# ============================================================

def process_batch(
    batch,
    model,
    client,
    collection_name,
    embedding_batch_size,
    config,
):

    texts = [
        item["text"]
        for item in batch
    ]

    # Append EOS to NV-Embed-v2 passages.
    if config["add_eos"]:

        eos_token = model.tokenizer.eos_token

        texts_to_encode = [
            text + eos_token
            for text in texts
        ]

    else:

        texts_to_encode = texts


    # document / passage embedding
    # Apply the query instruction only to queries during search.
    embeddings = model.encode(
        texts_to_encode,
        batch_size=embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    points = []

    for item, vector in zip(
        batch,
        embeddings
    ):

        point_id = make_point_id(
            patent_id=item["patent_id"],
            unit_type=item["unit_type"],
            unit_id=item["unit_id"],
        )

        payload = {

            "embedding_model":
                config["model_name"],

            "record_id":
                item["record_id"],

            "patent_id":
                item["patent_id"],

            "family_id":
                item["family_id"],

            "pub_date":
                item["pub_date"],

            "app_date":
                item["app_date"],

            "title":
                item["title"],

            "ipc_codes":
                item["ipc_codes"],

            "cpc_codes":
                item["cpc_codes"],

            "unit_type":
                item["unit_type"],

            "unit_id":
                item["unit_id"],

            "text":
                item["text"],

            "num_claims":
                item["num_claims"],

            "source_file":
                item["source_file"],
        }

        if item["unit_type"] == "claim":

            payload["claim_no"] = (
                item["claim_no"]
            )


        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            )
        )


    client.upsert(
        collection_name=collection_name,
        points=points,
    )

# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    config = MODEL_CONFIGS[args.model]

    collection_name = (
        args.collection
        if args.collection
        else config["collection_name"]
    )

    embedding_batch_size = (
        args.batch_size
        if args.batch_size is not None
        else config["default_batch_size"]
    )


    print("=" * 80)
    print(
        "Patent Multi-Granularity Vector DB Builder"
    )
    print("=" * 80)

    print(
        f"Data directory : "
        f"{args.data_dir}"
    )

    print(
        f"Pattern        : "
        f"{args.pattern}"
    )

    print(
        f"Qdrant path    : "
        f"{args.qdrant_path}"
    )

    print(
        f"Collection     : "
        f"{collection_name}"
    )

    print(f"Model key      : {args.model}")
    print(f"Collection     : {collection_name}")
    print(f"Embedding model: {config['model_name']}")
    print(f"Max seq length : {config['max_seq_length']}")
    print(f"Batch size     : {embedding_batch_size}")

    # ========================================================
    # Input files
    # ========================================================

    files = sorted(
        args.data_dir.glob(
            args.pattern
        )
    )


    if not files:

        raise FileNotFoundError(
            f"No files found:\n"
            f"{args.data_dir / args.pattern}"
        )


    print()
    print(
        f"Patent files found: "
        f"{len(files):,}"
    )


    # ========================================================
    # GPU
    # ========================================================

    device = get_device()


    # ========================================================
    # Qwen3 Embedding
    # ========================================================

    print()
    print(
        "Loading Qwen3 embedding model..."
    )

    model = load_embedding_model(
        config=config,
        device=device,
    )


    vector_size = (
        model
        .get_sentence_embedding_dimension()
    )


    print(
        f"Embedding dimension: "
        f"{vector_size}"
    )

    print(
        f"Model max seq length: "
        f"{model.max_seq_length}"
    )


    # ========================================================
    # Qdrant Local
    # ========================================================

    args.qdrant_path.mkdir(
        parents=True,
        exist_ok=True
    )


    client = QdrantClient(
        path=str(
            args.qdrant_path
        )
    )


    # ========================================================
    # Collection
    # ========================================================

    collection_exists = (
        client.collection_exists(
            collection_name
        )
    )


    if (
        collection_exists
        and args.recreate
    ):

        print()
        print(
            f"Deleting existing collection: "
            f"{collection_name}"
        )

        client.delete_collection(
            collection_name=
                collection_name
        )

        collection_exists = False


    if not collection_exists:

        print()
        print(
            f"Creating collection: "
            f"{collection_name}"
        )


        client.create_collection(

            collection_name=
                collection_name,

            vectors_config=
                models.VectorParams(

                    size=
                        vector_size,

                    distance=
                        models.Distance.COSINE,

                ),

        )

    else:

        print()
        print(
            f"Using existing collection: "
            f"{collection_name}"
        )


    # ========================================================
    # Build DB
    # ========================================================

    batch = []

    total_vectors = 0

    count_by_type = {

        "document": 0,
        "whole_claims": 0,
        "claim": 0,

    }


    for unit in iter_text_units(
        files
    ):

        batch.append(
            unit
        )


        if (
            len(batch)
            >= args.upsert_size
        ):

            process_batch(
                batch=batch,
                model=model,
                client=client,
                collection_name=collection_name,
                embedding_batch_size=embedding_batch_size,
                config=config,
            )


            for item in batch:

                count_by_type[
                    item["unit_type"]
                ] += 1


            total_vectors += len(
                batch
            )


            print(
                f"\rIndexed vectors: "
                f"{total_vectors:,}",
                end="",
                flush=True,
            )


            batch = []


    # Final batch.
    if batch:

        process_batch(
            batch=batch,
            model=model,
            client=client,
            collection_name=collection_name,
            embedding_batch_size=embedding_batch_size,
            config=config,
        )


        for item in batch:

            count_by_type[
                item["unit_type"]
            ] += 1


        total_vectors += len(
            batch
        )


    # ========================================================
    # Summary
    # ========================================================

    print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


    print(
        f"Patent files      : "
        f"{len(files):,}"
    )

    print(
        f"Document vectors  : "
        f"{count_by_type['document']:,}"
    )

    print(
        f"Whole claims      : "
        f"{count_by_type['whole_claims']:,}"
    )

    print(
        f"Claim vectors     : "
        f"{count_by_type['claim']:,}"
    )

    print(
        f"Total vectors     : "
        f"{total_vectors:,}"
    )

    print(
        f"Qdrant path       : "
        f"{args.qdrant_path}"
    )


if __name__ == "__main__":
    main()
