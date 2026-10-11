"""Qdrant 기반 특허/화학구조 벡터 검색 엔진.

모듈 구성
  config.py               설정(환경변수/JSON) - 비밀값은 환경변수만 사용
  client.py               클라이언트 팩토리, 컬렉션/alias 생성
  schemas.py              컬렉션 스키마, payload 인덱스, UUID5 포인트 ID 정책
  filters.py              시간/패밀리/자기제외 공통 필터 빌더
  patent_retriever.py     BM25 / dense / hybrid 검색
  structure_repository.py 구조 컬렉션 조회 + RDKit exact Tanimoto
  hybrid_fusion.py        RRF/DBSF 융합, 구조 재정렬 및 가중치 재정규화
"""
