"""선행기술 검색 CLI.

TARGET 특허번호(또는 자유 텍스트)로 PRIOR 특허 후보를 검색한다.
특허번호로 검색하면 그 특허의 출원일/패밀리를 이용해 시간·패밀리·자기제외 필터가
자동으로 적용된다. 자유 텍스트 검색은 필터 기준이 없으므로 --app-date/--family-id를
주지 않으면 명시적으로 --allow-unfiltered를 요구한다.

입력 필드(--field)
    색인은 입력 필드 조합별로 벡터를 따로 저장한다(claims_* / title_claims_*).
    BM25 상태 파일도 필드별로 따로 저장된다(bm25_state_{field}.json).
    static-vectors dense 인코더는 같은 필드의 BM25 IDF로 가중하므로, 필드가 어긋나면
    적재 때와 다른 벡터로 검색하게 된다. 그래서 이 CLI는
      (1) 선택한 필드의 상태 파일로 dense/BM25 인코더를 모두 초기화하고
      (2) 검색 전에 payload의 embedding_signature_by_field 와 현재 dense 서명을 대조해
          다르면 검색을 시작하지 않는다.

사용 예:
    python3 scripts/qdrant/qdrant_search.py --patent EP3660125A1 --mode hybrid --limit 10
    python3 scripts/qdrant/qdrant_search.py --patent EP3660125A1 --field claims --mode bm25
    python3 scripts/qdrant/qdrant_search.py --patent EP3660125A1 --mode hybrid --rerank-structure
    python3 scripts/qdrant/qdrant_search.py --text "polyolefin catalyst composition" \
        --app-date 20200101 --family-id 12345678 --mode bm25
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
    """검색을 시작하기 전에 발견된 문제(상태 파일 부재, 인코더 서명 불일치 등).

    이 예외가 나면 검색을 실행하지 않는다. 잘못된 인코더로 낸 결과를 정상 결과처럼
    출력하는 것이 가장 위험하기 때문이다.
    """


def bm25_state_path(field):
    """필드별 BM25 상태 파일 경로.

    scripts/qdrant/qdrant_index_patents.py / scripts/evaluation/evaluate_qdrant_retrieval.py 와 동일 규칙.
    (레거시 단일 파일 bm25_state.json 은 더 이상 쓰지 않는다 - 색인과 어긋난다.)
    """
    return os.path.join(DERIVED_DIR, f'bm25_state_{field}.json')


def load_field_encoders(settings, field, state_path=None):
    """선택한 필드 전용 (dense, sparse) 인코더를 만든다.

    dense(static-vectors)도 같은 필드의 BM25 IDF를 쓰므로 상태 파일이 없으면
    적재 때와 다른 벡터가 만들어진다. 따라서 없으면 검색을 시작하지 않는다.
    """
    if field not in TEXT_FIELDS:
        raise SearchSetupError(f'알 수 없는 텍스트 필드: {field} (가능: {sorted(TEXT_FIELDS)})')
    path = state_path or bm25_state_path(field)
    if not os.path.exists(path):
        raise SearchSetupError(
            f"'{field}' 필드의 BM25 상태 파일이 없습니다: {path}\n"
            f'  먼저 scripts/qdrant/qdrant_index_patents.py 를 실행해 필드별 상태 파일을 만드세요.')
    sparse = BM25SparseEncoder.load(path)
    dense = build_dense_encoder(settings, idf_state_path=path)
    return dense, sparse


def verify_encoder_signature(client, settings, field, dense_encoder):
    """적재 시 인코더 == 검색 시 인코더 확인. 다르면 검색하지 않는다.

    반환: {'field', 'indexed', 'search', 'match'}
    """
    try:
        points, _ = client.scroll(settings.patents_collection, limit=1, with_payload=True)
    except Exception as exc:
        raise SearchSetupError(
            f'{settings.patents_collection} 컬렉션을 읽을 수 없습니다: '
            f'{type(exc).__name__}: {exc}') from exc
    if not points:
        raise SearchSetupError(
            f'{settings.patents_collection} 컬렉션이 비어 있습니다 - 먼저 색인이 필요합니다.')

    stored_by_field = (points[0].payload or {}).get('embedding_signature_by_field') or {}
    stored = stored_by_field.get(field)
    current = dense_encoder.signature()
    if not stored:
        raise SearchSetupError(
            f"적재된 포인트에 '{field}' 필드의 인코더 서명(embedding_signature_by_field)이 "
            f'없습니다. 필드별 벡터가 없는 구버전 색인으로 보입니다 - '
            f'scripts/qdrant/qdrant_index_patents.py 로 재색인하세요. '
            f'(payload에 있는 필드: {sorted(stored_by_field) or "없음"})')
    if stored != current:
        raise SearchSetupError(
            f'인코더 불일치({field}): 적재 {stored} != 검색 {current}\n'
            f'  검색 dense: {dense_encoder.describe()}\n'
            f'  BM25 상태 : {bm25_state_path(field)}\n'
            f'  색인에 쓴 설정(CHEMRAG_DENSE_BACKEND/CHEMRAG_DENSE_MODEL 등)으로 맞추거나 '
            f'재색인이 필요합니다.')
    return {'field': field, 'indexed': stored, 'search': current, 'match': True}


def apply_structure_rerank(client, settings, hits, query_patent_id):
    """상위 후보에 RDKit exact Tanimoto 재정렬. (hits, stats) 반환.

    구조가 없으면 0이 아니라 None으로 처리해 텍스트 점수만 쓰게 한다.
    """
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
    src.add_argument('--patent', help='쿼리로 쓸 TARGET 특허번호(예: EP3660125A1)')
    src.add_argument('--text', help='자유 텍스트 질의')
    ap.add_argument('--mode', choices=('bm25', 'dense', 'hybrid'), default='hybrid')
    ap.add_argument('--field', choices=tuple(sorted(TEXT_FIELDS)), default=DEFAULT_FIELD,
                    help='검색에 쓸 입력 텍스트 필드(색인된 필드별 벡터를 선택). '
                         f'기본 {DEFAULT_FIELD}')
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--app-date', help='자유 텍스트 검색 시 시간 필터 기준일(YYYYMMDD)')
    ap.add_argument('--family-id', help='자유 텍스트 검색 시 제외할 patent family')
    ap.add_argument('--rerank-structure', action='store_true',
                    help='상위 후보에 RDKit exact Tanimoto 재정렬 적용')
    ap.add_argument('--allow-unfiltered', action='store_true',
                    help='시간/패밀리 필터 없이 검색(디버그 전용)')
    ap.add_argument('--json', action='store_true', help='JSON으로 출력')
    return ap


def main():
    args = build_arg_parser().parse_args()

    settings = load_settings()
    client = get_client(settings)

    # 검색 전 점검: 필드별 상태 파일 + 적재/검색 인코더 서명 일치.
    # 하나라도 어긋나면 결과를 내지 않고 중단한다.
    try:
        dense, sparse = load_field_encoders(settings, args.field)
        sig_check = verify_encoder_signature(client, settings, args.field, dense)
    except SearchSetupError as exc:
        raise SystemExit(f'[중단] {exc}')

    retriever = PatentRetriever(client, settings, {args.field: dense}, {args.field: sparse},
                                default_field=args.field)

    if args.patent:
        payload = retriever.get_patent_payload(args.patent)
        if not payload:
            raise SystemExit(f'{args.patent} 를 컬렉션에서 찾을 수 없습니다')
        query_text, missing = build_field_text(payload.get('title'), payload.get('claims_text'),
                                               args.field)
        if missing:
            print(f'[경고] {args.patent} 의 claims_text 가 비어 있습니다 '
                  f'(field={args.field} 질의 텍스트 길이 {len(query_text)})')
        pf = build_prior_art_filter(args.patent, payload.get('app_date'), payload.get('family_id'))
    else:
        query_text = args.text
        if not (args.app_date or args.family_id) and not args.allow_unfiltered:
            raise SystemExit('자유 텍스트 검색에는 --app-date/--family-id 로 필터 기준을 주거나 '
                             '--allow-unfiltered 를 명시해야 합니다.')
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

    print(f'질의: {args.patent or args.text!r} | 모드: {args.mode} | 필드: {args.field} '
          f'| 검색 {latency_ms:.1f}ms')
    print(f'  인코더 서명 일치: {sig_check["indexed"]} (dense={dense.describe()["model"]})')
    if pf:
        for rule in pf.describe()['rules']:
            print(f'  필터: {rule}')
    else:
        print('  필터: 없음(--allow-unfiltered)')
    if rerank_stats:
        print(f'  구조 재정렬: 상위 {rerank_stats["reranked"]}건 중 구조 보유 {rerank_stats["with_structure"]}건')
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
