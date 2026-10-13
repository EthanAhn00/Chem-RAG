"""Collection schemas, payload indexes, and point-ID policy.
Never use Python hash() for IDs: values vary across processes/PYTHONHASHSEED. Use a fixed UUID5 namespace so identical inputs always produce identical IDs and idempotent upserts.
"""
import uuid
from qdrant_client import models

from .config import ID_NAMESPACE

_NS = uuid.UUID(ID_NAMESPACE)


def patent_point_id(patent_id):
    """Patent point ID: UUID5 based on patent_id."""
    if not patent_id:
        raise ValueError('patent_id is empty')
    return str(uuid.uuid5(_NS, f'patent:{patent_id}'))


def structure_point_id(patent_id, mol_id):
    """Structure point ID: UUID5 based on (patent_id, mol_id)."""
    if not patent_id or not mol_id:
        raise ValueError('patent_id/mol_id is empty')
    return str(uuid.uuid5(_NS, f'structure:{patent_id}:{mol_id}'))


# Patent collection payload indexes.
# Embedded/local Qdrant does not actually create payload indexes; it warns
# and uses brute-force scans. Indexes take effect only in server mode.
# Declare them through the same code; verification reports actual creation.
PATENT_PAYLOAD_INDEXES = [
    ('patent_id', models.PayloadSchemaType.KEYWORD),
    ('pub_date', models.PayloadSchemaType.KEYWORD),
    ('app_date', models.PayloadSchemaType.KEYWORD),
    ('family_id', models.PayloadSchemaType.KEYWORD),
    ('ipc_codes', models.PayloadSchemaType.KEYWORD),
    ('ipc_sections', models.PayloadSchemaType.KEYWORD),
    ('role', models.PayloadSchemaType.KEYWORD),
    ('split', models.PayloadSchemaType.KEYWORD),
    ('has_structure', models.PayloadSchemaType.BOOL),
    ('missing_claims', models.PayloadSchemaType.BOOL),
    ('pub_date_int', models.PayloadSchemaType.INTEGER),
    ('app_date_int', models.PayloadSchemaType.INTEGER),
]

STRUCTURE_PAYLOAD_INDEXES = [
    ('patent_id', models.PayloadSchemaType.KEYWORD),
    ('mol_id', models.PayloadSchemaType.KEYWORD),
    ('inchikey', models.PayloadSchemaType.KEYWORD),
    ('source', models.PayloadSchemaType.KEYWORD),
    ('in_claim', models.PayloadSchemaType.KEYWORD),
    ('in_figure', models.PayloadSchemaType.KEYWORD),
    ('location_status', models.PayloadSchemaType.KEYWORD),
]

# Three-valued structure-location logic: unverified means unknown, not false.
TRUE = 'true'
FALSE = 'false'
UNKNOWN = 'unknown'
TRISTATE = (TRUE, FALSE, UNKNOWN)


# Store separate named vectors for both input-field combinations
# to separate algorithm effects from input-field effects.
TEXT_FIELDS = {
    'claims': {
        'dense': 'claims_dense',
        'sparse': 'claims_bm25',
        'description': 'claims_text only',
    },
    'title_claims': {
        'dense': 'title_claims_dense',
        'sparse': 'title_claims_bm25',
        'description': 'title + claims_text',
    },
}


def build_field_text(title, claims_text, field):
    """Build embedding text for each field combination; shared by all indexing/search paths. Return (text, missing_claims). With no claims_text, retain title-only text and mark missing_claims=True rather than discarding the patent."""
    title = (title or '').strip()
    claims = (claims_text or '').strip()
    if field == 'claims':
        return claims, (not claims)
    if field == 'title_claims':
        if claims:
            return (f'{title}\n{claims}' if title else claims), False
        return title, True
    raise ValueError(f'Unknown text field: {field} (available: {sorted(TEXT_FIELDS)})')


def patent_vectors_config(dense_dim):
    return {spec['dense']: models.VectorParams(size=int(dense_dim), distance=models.Distance.COSINE)
            for spec in TEXT_FIELDS.values()}


def patent_sparse_config():
    # BM25 weights are computed on the client; do not use the Qdrant IDF modifier
    # or IDF would be applied twice. See embeddings/sparse_encoder.py.
    return {spec['sparse']: models.SparseVectorParams() for spec in TEXT_FIELDS.values()}


def structure_sparse_config():
    # Store Morgan ECFP4 2048-bit on-bit indexes as sparse vectors.
    # Dot-product scores are not Tanimoto; use them only for candidate generation
    # and recompute actual similarity with RDKit exact Tanimoto.
    return {'morgan_ecfp4_2048': models.SparseVectorParams()}
