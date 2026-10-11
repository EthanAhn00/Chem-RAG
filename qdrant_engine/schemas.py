"""컬렉션 스키마, payload 인덱스 정의, 포인트 ID 정책.

ID 정책(중요): Python 내장 hash()는 프로세스마다 값이 달라지므로(PYTHONHASHSEED)
절대 쓰지 않는다. 모든 포인트 ID는 고정 네임스페이스 기반 UUID5로 만들어
같은 입력이면 언제 실행해도 같은 ID가 나오고 upsert가 멱등이 되게 한다.
"""
import uuid
from qdrant_client import models

from .config import ID_NAMESPACE

_NS = uuid.UUID(ID_NAMESPACE)


def patent_point_id(patent_id):
    """특허 포인트 ID - patent_id 기반 UUID5."""
    if not patent_id:
        raise ValueError('patent_id가 비어 있습니다')
    return str(uuid.uuid5(_NS, f'patent:{patent_id}'))


def structure_point_id(patent_id, mol_id):
    """구조 포인트 ID - (patent_id, mol_id) 기반 UUID5."""
    if not patent_id or not mol_id:
        raise ValueError('patent_id/mol_id가 비어 있습니다')
    return str(uuid.uuid5(_NS, f'structure:{patent_id}:{mol_id}'))


# 특허 컬렉션 payload 인덱스.
# 주의: 임베디드(local) 모드에서는 Qdrant가 payload 인덱스를 실제로 만들지 않고
# 경고만 낸다(브루트포스 스캔). 서버 모드에서만 실제 효과가 있으며, 여기서는
# 어느 쪽이든 같은 코드로 선언해두고 verify 스크립트가 실제 생성 여부를 보고한다.
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

# 구조 위치 판정에 쓰는 3-값 논리. 데이터로 확인되지 않으면 false가 아니라 unknown.
TRUE = 'true'
FALSE = 'false'
UNKNOWN = 'unknown'
TRISTATE = (TRUE, FALSE, UNKNOWN)


# 입력 텍스트 필드 조합. 알고리즘 효과와 입력 필드 효과를 분리해 평가하기 위해
# 두 조합을 각각 별도의 named vector로 저장한다.
TEXT_FIELDS = {
    'claims': {
        'dense': 'claims_dense',
        'sparse': 'claims_bm25',
        'description': 'claims_text 만 사용',
    },
    'title_claims': {
        'dense': 'title_claims_dense',
        'sparse': 'title_claims_bm25',
        'description': 'title + claims_text 사용',
    },
}


def build_field_text(title, claims_text, field):
    """필드 조합별 임베딩 입력 텍스트. 모든 색인/검색 경로가 이 함수 하나만 쓴다.

    반환: (text, missing_claims)
    claims_text가 없으면 특허를 버리지 않고 title만 쓰며 missing_claims=True로 표시한다.
    """
    title = (title or '').strip()
    claims = (claims_text or '').strip()
    if field == 'claims':
        return claims, (not claims)
    if field == 'title_claims':
        if claims:
            return (f'{title}\n{claims}' if title else claims), False
        return title, True
    raise ValueError(f'알 수 없는 텍스트 필드: {field} (가능: {sorted(TEXT_FIELDS)})')


def patent_vectors_config(dense_dim):
    return {spec['dense']: models.VectorParams(size=int(dense_dim), distance=models.Distance.COSINE)
            for spec in TEXT_FIELDS.values()}


def patent_sparse_config():
    # BM25 가중치를 클라이언트에서 직접 계산해 넣으므로 Qdrant IDF modifier는 쓰지 않는다
    # (쓰면 IDF가 두 번 곱해진다). embeddings/sparse_encoder.py 참고.
    return {spec['sparse']: models.SparseVectorParams() for spec in TEXT_FIELDS.values()}


def structure_sparse_config():
    # Morgan ECFP4 2048-bit의 on-bit 인덱스를 sparse 벡터로 저장.
    # 주의: 이 벡터의 dot-product 점수는 Tanimoto가 아니다. 후보 생성용일 뿐이고
    # 실제 유사도는 RDKit exact Tanimoto로 다시 계산한다.
    return {'morgan_ecfp4_2048': models.SparseVectorParams()}
