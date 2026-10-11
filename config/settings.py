import os

# 프로젝트 설정 및 디렉토리 정의
# 기본 경로, IPC Section C 범위, 동적 멀티모달 가중치 파라미터 제공

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DRIVE_DIR = os.environ.get(
    'CHEMRAG_EXTERNAL_DATA_ROOT', os.path.join(PROJECT_DIR, 'data', 'external')
)
EPO_JSON_DIR = os.environ.get(
    'CHEMRAG_EPO_JSON_DIR',
    os.path.join(EXTERNAL_DRIVE_DIR, 'EPO_SR_parsed_260207', 'Simple'),
)
EPO_XML_DIR = os.environ.get(
    'CHEMRAG_EPO_XML_DIR',
    os.path.join(EXTERNAL_DRIVE_DIR, 'EPO_priorEPO_matched_xml_only'),
)

# 마스터 ID -> 특허번호 / XML 파일명 맵 (id, T_file, P_file, T_id, P_id, T_IPC, P_IPC)
# Simple/*.json 레코드 자체는 실제 EP 공개번호 없이 순차 파일 id만 보유
# EPO_SR_parsed_260207/260207_Target_data.xlsx로부터 이 매핑을 별도 복원
# 이 매핑 없이는 레코드를 XML 도면 파일이나 SureChEMBL/PatCID와 특허번호 기준으로 조인 불가
ID_PATENT_MAP_PATH = os.path.join(PROJECT_DIR, 'data', 'reference', 'id_patent_map.csv')

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'samples')
BENCHMARK_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'benchmark', 'results')

# scripts/db_export/build_time_family_split.py의 레거시 connected-record/time split.
# family_id 기반 분할 또는 최종 T123 five-fold assignment와 동일하지 않다.
# 레코드 파일명 -> 'train' | 'test' | 'train_unresolved' 매핑. 선택 사항: 이 파일이 없어도
# 파이프라인은 정상 동작(전체 셋 지표만 계산)
TRAIN_TEST_SPLIT_PATH = os.path.join(DATA_DIR, 'reference', 'train_test_split.json')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
os.makedirs(BENCHMARK_OUTPUT_DIR, exist_ok=True)

# IPC Section C(화학·야금: C01 ~ C99) 및 의약 서브클래스(A61K, A61P)
SECTION_C_PREFIXES = ['C', 'A61K', 'A61P']

# 하이브리드 RAG 점수 가중치(단일 소스 - 설계 문서의
# Score(Q,D) = alpha*TextSim + gamma*TanimotoSim + beta*VisionBonus 공식과 일치, alpha+gamma+beta = 1.0)
# 화학구조(SMILES)와 도면이 둘 다 사용 가능할 때 쓰는 가중치. 한쪽 또는 둘 다 사용 불가능하면
# MultimodalChemicalRanker가 사용 가능한 성분들만으로 재정규화(rag_engine/multimodal_ranker.py 참고)
# - 조용히 기본값 처리하거나 점수를 지어내지 않음
DEFAULT_TEXT_WEIGHT = 0.5
DEFAULT_TANIMOTO_WEIGHT = 0.3
DEFAULT_VISION_WEIGHT = 0.2

MAX_STREAM_SAMPLES = 2000

# ---- EPO OPS(Open Patent Services) API 인증 정보 ----
# 로컬 XML 코퍼스(EPO_XML_DIR)에는 <img file="imgf0001.tif"/> 같은 참조만 있고
# 실제 도면 래스터 이미지(.tif)는 포함돼 있지 않음(원본 대용량 아카이브 미보유).
# 이 때문에 공개 도면 이미지를 EPO OPS API로 개별 조회해서 보충 - OCSR(vision_ocsr/ocsr_pipeline.py)의
# 이미지 소스로 사용됨. 반드시 환경변수로 설정할 것 (.env 등 - 저장소에는 커밋하지 말 것)
EPO_OPS_CONSUMER_KEY = os.environ.get('EPO_OPS_CONSUMER_KEY')
EPO_OPS_CONSUMER_SECRET = os.environ.get('EPO_OPS_CONSUMER_SECRET')

# OCSR(도면->SMILES) 파이프라인 온/오프 스위치. 기본 False - SureChEMBL 매칭이 실패한 레코드에
# 한해 OPS로 도면을 받아와 DECIMER로 구조를 인식하는 폴백 경로라 OPS 쿼터/DECIMER 설치가
# 준비된 뒤에만 True로 켤 것 (run_chem_rag.py 참고)
ENABLE_OCSR_FALLBACK = os.environ.get('ENABLE_OCSR_FALLBACK', '0') == '1'

# OPS에서 받은 원본 페이지 이미지, 크롭 결과, OCSR 인식 결과를 캐싱해 동일 특허 재조회 시
# API 쿼터를 다시 쓰지 않도록 함
OCSR_CACHE_DIR = os.path.join(DATA_DIR, 'ocsr_cache')
os.makedirs(OCSR_CACHE_DIR, exist_ok=True)

# 특허 1건당 OPS에서 받아올 최대 도면 페이지 수 (무료 등급 주당 3.5GB 트래픽 보호용 상한)
OCSR_MAX_PAGES_PER_PATENT = 3
