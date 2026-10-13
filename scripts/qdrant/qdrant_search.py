"""Prior-art search CLI using a TARGET patent number or free text. Patent-number queries apply filing-date/family/self-exclusion filters automatically. Free text requires --app-date/--family-id or explicit --allow-unfiltered.
Field-specific named vectors (claims_* / title_claims_*) and BM25 state files (bm25_state_{field}.json) separate inputs. static-vectors uses the same field's BM25 IDF; mismatched fields change vectors. Initialize both encoders from the selected field's state and compare payload embedding_signature_by_field with the current dense signature before search; stop on mismatch.
Examples: --patent EP3660125A1 --mode hybrid --limit 10; --field claims --mode bm25; --rerank-structure; or --text 'polyolefin catalyst composition' --app-date 20200101 --family-id 12345678 --mode bm25.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import DATA_DIR
from qdrant_engine.config import load_settings
from qdrant_engine.client import get_client
from qdrant_engine.filters import build_prior_art_filter
from qdrant_engine.patent_retriever import PatentRetriever
from qdrant_engine.schemas import TEXT_FIELDS, build_field_text
from qdrant_engine.structure_repository import StructureRepository, max_tanimoto
from qdrant_engine.hybrid_fusion import rerank_with_structure
from embeddings.registry import build_dense_encoder
from embeddings.sparse_encoder import BM25SparseEncoder

DERIVED_DIR = os.path.join(DATA_DIR, 'db_export', 'derived')
DEFAULT_FIELD = 'title_claims'


class SearchSetupError(RuntimeError):
    """Pre-search failure such as missing state or mismatched encoder signature. Never execute search or present results from an incompatible encoder as valid."""


def bm25_state_path(field):
    """Per-field BM25 state path matching qdrant_index_patents.py and evaluate_qdrant_retrieval.py. Do not use legacy bm25_state.json, which disagrees with field-specific indexing."""
    return os.path.join(DERIVED_DIR, f'bm25_state_{field}.json')


def load_field_encoders(settings, field, state_path=None):
    """Build dense/sparse encoders for the selected field. static-vectors depends on that field's BM25 IDF; without state it would differ from the indexed encoder, so stop before search."""
    if field not in TEXT_FIELDS:
        raise SearchSetupError(f'Unknown text field: {field} (available: {sorted(TEXT_FIELDS)})')
    path = state_path or bm25_state_path(field)
    if not os.path.exists(path):
        raise SearchSetupError(
            f"'{field}' field: BM25 state file is missing: {path}\n"
            f'  Run scripts/qdrant/qdrant_index_patents.py first to create field-specific state files.')
    sparse = BM25SparseEncoder.load(path)
    dense = build_dense_encoder(settings, idf_state_path=path)
    return dense, sparse


def verify_encoder_signature(client, settings, field, dense_encoder):
    """Verify indexing encoder == search encoder and stop on mismatch. Return {'field', 'indexed', 'search', 'match'}."""
    try:
        points, _ = client.scroll(settings.patents_collection, limit=1, with_payload=True)
    except Exception as exc:
        raise SearchSetupError(
            f'{settings.patents_collection} collection cannot be read: '
            f'{type(exc).__name__}: {exc}') from exc
    if not points:
        raise SearchSetupError(
            f'{settings.patents_collection} collection is empty; indexing is required first.')

    stored_by_field = (points[0].payload or {}).get('embedding_signature_by_field') or {}
    stored = stored_by_field.get(field)
    current = dense_encoder.signature()
    if not stored:
        raise SearchSetupError(
            f"Indexed points lack the '{field}' field encoder signature (embedding_signature_by_field); "
            f'this appears to be a legacy index without field-specific vectors. '
            f'Reindex with scripts/qdrant/qdrant_index_patents.py. '
            f'(payload fields: {sorted(stored_by_field) or "none"})')
    if stored != current:
        raise SearchSetupError(
            f'Encoder mismatch ({field}): indexed {stored} != search {current}\n'
            f'  Search dense: {dense_encoder.describe()}\n'
            f'  BM25 state: {bm25_state_path(field)}\n'
            f'  Match indexing settings (CHEMRAG_DENSE_BACKEND/CHEMRAG_DENSE_MODEL etc.) '
            f'or reindex.')
    return {'field': field, 'indexed': stored, 'search': current, 'match': True}


def apply_structure_rerank(client, settings, hits, query_patent_id):
    """Rerank top candidates using RDKit exact Tanimoto; return (hits, stats). Missing structures are None, not zero, retaining text-only scores."""
    repo = StructureRepository(client, settings)
    smiles_index = repo.load_smiles_index()
    q_fps = repo.fingerprints_for_patent(query_patent_id, smiles_index) if query_patent_id else []

    def tanimoto_fn(cand_id):
        if not q_fps:
            return None
        cf = repo.fingerprints_for_patent(cand_id, smiles_index)
        return max_tanimoto(q_fps, cf) if cf else None

    return rerank_with_structure(hits, tanimoto_fn, settings.text_weight,
                                 settings.structure_rerank_weight,
                                 settings.structure_rerank_top_k)


def build_arg_parser():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--patent', help='TARGET patent number for the query (e.g. EP3660125A1)')
    src.add_argument('--text', help='Free-text query')
    ap.add_argument('--mode', choices=('bm25', 'dense', 'hybrid'), default='hybrid')
    ap.add_argument('--field', choices=tuple(sorted(TEXT_FIELDS)), default=DEFAULT_FIELD,
                    help='Input text field selecting the indexed field-specific vectors. '
                         f'Default {DEFAULT_FIELD}')
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--app-date', help='Time-filter reference date for free-text queries (YYYYMMDD)')
    ap.add_argument('--family-id', help='Patent family to exclude for free-text queries')
    ap.add_argument('--rerank-structure', action='store_true',
                    help='Apply RDKit exact Tanimoto reranking to top candidates')
    ap.add_argument('--allow-unfiltered', action='store_true',
                    help='Search without time/family filtering (debug only)')
    ap.add_argument('--json', action='store_true', help='Print JSON')
    return ap


def main():
    args = build_arg_parser().parse_args()

    settings = load_settings()
    client = get_client(settings)

    # Pre-search checks: per-field state and matching indexing/search encoder signatures.
    # Stop without results if either check fails.
    try:
        dense, sparse = load_field_encoders(settings, args.field)
        sig_check = verify_encoder_signature(client, settings, args.field, dense)
    except SearchSetupError as exc:
        raise SystemExit(f'[STOP] {exc}')

    retriever = PatentRetriever(client, settings, {args.field: dense}, {args.field: sparse},
                                default_field=args.field)

    if args.patent:
        payload = retriever.get_patent_payload(args.patent)
        if not payload:
            raise SystemExit(f'{args.patent} not found in the collection')
        query_text, missing = build_field_text(payload.get('title'), payload.get('claims_text'),
                                               args.field)
        if missing:
            print(f'[WARNING] {args.patent} has empty claims_text '
                  f'(field={args.field} query text length {len(query_text)})')
        pf = build_prior_art_filter(args.patent, payload.get('app_date'), payload.get('family_id'))
    else:
        query_text = args.text
        if not (args.app_date or args.family_id) and not args.allow_unfiltered:
            raise SystemExit('Free-text queries require --app-date/--family-id filter references or '
                             'explicit --allow-unfiltered.')
        pf = None if args.allow_unfiltered and not (args.app_date or args.family_id) else \
            build_prior_art_filter('__query__', args.app_date, args.family_id)

    hits, latency_ms = retriever.search(args.mode, query_text, pf,
                                        limit=max(args.limit, settings.structure_rerank_top_k
                                                  if args.rerank_structure else args.limit),
                                        field=args.field,
                                        allow_unfiltered=args.allow_unfiltered)

    rerank_stats = None
    if args.rerank_structure:
        hits, rerank_stats = apply_structure_rerank(client, settings, hits, args.patent)

    hits = hits[: args.limit]
    if args.json:
        print(json.dumps({'query': args.patent or args.text, 'mode': args.mode,
                          'field': args.field,
                          'vector_names': TEXT_FIELDS[args.field],
                          'bm25_state': bm25_state_path(args.field),
                          'encoder_signature': sig_check,
                          'latency_ms': round(latency_ms, 2),
                          'filter': pf.describe() if pf else None,
                          'rerank': rerank_stats, 'hits': hits}, ensure_ascii=False, indent=2))
        return

    print(f'Query: {args.patent or args.text!r} | mode: {args.mode} | field: {args.field} '
          f'| retrieved {latency_ms:.1f}ms')
    print(f'  Encoder signatures match: {sig_check["indexed"]} (dense={dense.describe()["model"]})')
    if pf:
        for rule in pf.describe()['rules']:
            print(f'  Filter: {rule}')
    else:
        print('  Filter: none (--allow-unfiltered)')
    if rerank_stats:
        print(f'  Structure reranking: top {rerank_stats["reranked"]}; structure-bearing {rerank_stats["with_structure"]} records')
    print()
    for i, h in enumerate(hits, 1):
        pl = h['payload']
        extra = ''
        if h.get('tanimoto_score') is not None:
            extra = f" | Tanimoto={h['tanimoto_score']}"
        print(f'{i:3d}. {pl.get("patent_id")}  score={h["score"]:.4f}  '
              f'pub={pl.get("pub_date")} family={pl.get("family_id")} '
              f'role={pl.get("role")}{extra}')


if __name__ == '__main__':
    main()
