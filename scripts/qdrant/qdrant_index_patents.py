"""Index 4,123 patents into chemrag_patents_v1. Store field-specific dense/sparse vectors to separate algorithm and input effects:
claims_dense/claims_bm25 <- claims_text; title_claims_dense/title_claims_bm25 <- title+claims_text.
Keep title-only patents with missing_claims=true when claims are absent. Use patent_id UUID5 for idempotent upserts. Record embedding_backend/model/version/signature, per-field text_hash, and source_version in payloads.
Usage: python3 scripts/qdrant/qdrant_index_patents.py [--dense-backend static-vectors --recreate].
"""
import os
import sys
import csv
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
csv.field_size_limit(10 ** 9)

from qdrant_client import models

from config.settings import DATA_DIR
from qdrant_engine.config import load_settings
from qdrant_engine.client import get_client, ensure_patent_collection, is_local_mode
from qdrant_engine.schemas import patent_point_id, TEXT_FIELDS, build_field_text
from qdrant_engine.provenance import source_version, runtime_provenance
from embeddings.registry import build_dense_encoder, build_sparse_encoder
from embeddings.base import text_hash

DB_EXPORT_DIR = os.path.join(DATA_DIR, 'db_export')
DERIVED_DIR = os.path.join(DB_EXPORT_DIR, 'derived')


def bm25_state_path(field):
    return os.path.join(DERIVED_DIR, f'bm25_state_{field}.json')


def load_csv(name):
    with open(os.path.join(DB_EXPORT_DIR, f'{name}.csv'), 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def to_int_or_none(v):
    s = str(v or '').strip()
    return int(s) if s.isdigit() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense-backend', default=None)
    ap.add_argument('--dense-model', default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None, help='Debug limit on patents to index')
    ap.add_argument('--recreate', action='store_true')
    args = ap.parse_args()

    settings = load_settings(overrides={'dense_backend': args.dense_backend,
                                        'dense_model': args.dense_model,
                                        'batch_size': args.batch_size})
    os.makedirs(DERIVED_DIR, exist_ok=True)

    src_paths = [os.path.join(DB_EXPORT_DIR, f) for f in
                 ('patent_content.csv', 'split.csv', 'patent_molecule.csv')]
    src_version, src_detail = source_version(src_paths)
    print(f'[source_version] {src_version}')

    print('[1/6] Loading CSVs...')
    patents = load_csv('patent_content')
    split_rows = {r['id']: r for r in load_csv('split')}
    structure_patents = {r['patent_id'] for r in load_csv('patent_molecule')}
    if args.limit:
        patents = patents[: args.limit]
    print(f'  patent_content {len(patents)}; split {len(split_rows)}; structure-bearing patents {len(structure_patents)} records')

    # Prepare text for each field.
    field_texts = {field: [] for field in TEXT_FIELDS}
    metas = []
    csv_flag_mismatch = 0
    for r in patents:
        per_field = {}
        for field in TEXT_FIELDS:
            text, missing = build_field_text(r.get('title'), r.get('claims_text'), field)
            field_texts[field].append(text)
            per_field[field] = (text, missing)
        sp = split_rows.get(r['patent_id'], {})
        ipc_raw = (r.get('ipc_codes') or '').strip()
        ipc_list = [c for c in ipc_raw.split('|') if c]
        csv_flag = str(r.get('has_structure', '')).strip().lower() == 'true'
        actual_has_structure = r['patent_id'] in structure_patents
        if csv_flag != actual_has_structure:
            csv_flag_mismatch += 1
        metas.append({
            'patent_id': r['patent_id'],
            'title': r.get('title') or '',
            'claims_text': r.get('claims_text') or '',
            'pub_date': r.get('pub_date') or '',
            'app_date': r.get('app_date') or '',
            'pub_date_int': to_int_or_none(r.get('pub_date')),
            'app_date_int': to_int_or_none(r.get('app_date')),
            'family_id': str(r.get('family_id') or ''),
            'ipc_codes': ipc_list,
            'ipc_codes_raw': ipc_raw,
            'ipc_sections': sorted({c[0] for c in ipc_list if c}),
            'role': sp.get('role') or None,
            'split': sp.get('split') or None,
            'time_cutoff': sp.get('time_cutoff') or None,
            'has_structure': actual_has_structure,
            'has_structure_csv_flag': csv_flag,
            # Whether claims_text itself is empty, independent of the selected field combination.
            'missing_claims': not (r.get('claims_text') or '').strip(),
            'text_hash_claims': text_hash(per_field['claims'][0]),
            'text_hash_title_claims': text_hash(per_field['title_claims'][0]),
            'text_hash': text_hash(per_field['title_claims'][0]),   # Based on the default field.
            'source_version': src_version,
        })

    missing_claims_count = sum(1 for m in metas if m['missing_claims'])
    print(f'  Patents without claims_text (title-only embeddings): {missing_claims_count} records')
    print(f'  Mismatches between has_structure CSV flags and actual patent_molecule links: {csv_flag_mismatch} records')

    print('[2/6] Tokenizing and fitting BM25 per field...')
    from rag_engine.chemical_tokenizer import ChemicalTokenizer
    tokenizer = ChemicalTokenizer()
    field_tokens = {}
    sparse_encoders = {}
    for field in TEXT_FIELDS:
        t_tok = time.time()
        field_tokens[field] = [tokenizer.tokenize(t or '') for t in field_texts[field]]
        enc = build_sparse_encoder(settings)
        enc.fit(field_texts[field], token_lists=field_tokens[field])
        enc.save(bm25_state_path(field))
        sparse_encoders[field] = enc
        print(f'  {field}: tokenized {time.time()-t_tok:.1f}s, {enc.describe()}')

    print('[3/6] Preparing dense encoders per field...')
    dense_encoders = {}
    for field in TEXT_FIELDS:
        # static-vectors uses the same field's BM25 IDF weights.
        dense_encoders[field] = build_dense_encoder(settings, idf_state_path=bm25_state_path(field))
        print(f'  {field}: {dense_encoders[field].describe()}')

    dims = {f: e.dim for f, e in dense_encoders.items()}
    if len(set(dims.values())) != 1:
        raise SystemExit(f'Dense dimensions differ across fields: {dims}')
    dense_dim = next(iter(dims.values()))

    print('[4/6] Preparing collections...')
    client = get_client(settings)
    coll_report = ensure_patent_collection(client, settings, dense_dim, recreate=args.recreate)
    print(f'  {coll_report["collection"]} (mode={"local" if is_local_mode(settings) else "server"}) dim={dense_dim}')

    # Compute signatures/vector names once rather than for each point.
    signature_by_field = {f: dense_encoders[f].signature() for f in TEXT_FIELDS}
    vector_field_names = {f: dict(TEXT_FIELDS[f]) for f in TEXT_FIELDS}
    print(f'  Encoder signatures: {signature_by_field}')

    print('[5/6] Indexing...')
    t0 = time.time()
    batch = settings.batch_size
    total = 0
    for start in range(0, len(metas), batch):
        chunk_meta = metas[start:start + batch]
        vectors_per_field = {}
        for field, spec in TEXT_FIELDS.items():
            texts = field_texts[field][start:start + batch]
            toks = field_tokens[field][start:start + batch]
            vectors_per_field[field] = (
                dense_encoders[field].encode(texts, token_lists=toks),
                sparse_encoders[field].encode_documents(texts, token_lists=toks),
            )

        points = []
        for i, meta in enumerate(chunk_meta):
            payload = dict(meta)
            payload['embedding_backend'] = dense_encoders['title_claims'].name
            payload['embedding_model'] = dense_encoders['title_claims'].model_id
            payload['embedding_version'] = settings.embedding_version
            payload['embedding_dim'] = dense_dim
            payload['embedding_signature'] = signature_by_field['title_claims']
            payload['embedding_signature_by_field'] = signature_by_field
            payload['vector_fields'] = vector_field_names

            vector = {}
            for field, spec in TEXT_FIELDS.items():
                dvecs, svecs = vectors_per_field[field]
                vector[spec['dense']] = dvecs[i]
                idx, val = svecs[i]
                if idx:
                    vector[spec['sparse']] = models.SparseVector(indices=idx, values=val)
            points.append(models.PointStruct(
                id=patent_point_id(meta['patent_id']), vector=vector, payload=payload))
        client.upsert(settings.patents_collection, points=points, wait=True)
        total += len(points)
        if (start // batch) % 8 == 0:
            print(f'  {total}/{len(metas)} indexed ({time.time() - t0:.1f}s)')

    count = client.count(settings.patents_collection).count
    summary = {
        'provenance': runtime_provenance(settings, command=' '.join(sys.argv)),
        'source_version': src_version, 'source_files': src_detail,
        'input_patents': len(metas), 'upserted': total, 'collection_count': count,
        'collection': settings.patents_collection,
        'missing_claims': missing_claims_count,
        'has_structure_flag_mismatch_vs_patent_molecule': csv_flag_mismatch,
        'dense_dim': dense_dim,
        'dense_by_field': {f: e.describe() for f, e in dense_encoders.items()},
        'sparse_by_field': {f: e.describe() for f, e in sparse_encoders.items()},
        'vector_names': {f: TEXT_FIELDS[f] for f in TEXT_FIELDS},
        'elapsed_seconds': round(time.time() - t0, 1),
    }
    out = os.path.join(DERIVED_DIR, 'qdrant_index_patents_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('[6/6] Complete')
    print(json.dumps({k: v for k, v in summary.items() if k != 'provenance'}, ensure_ascii=False, indent=2))
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
