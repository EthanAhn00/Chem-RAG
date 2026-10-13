import os

# Project configuration and directory definitions.
# Default paths, IPC Section C scope, and dynamic multimodal weights.

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

# Master ID -> patent number / XML filename map (id, T_file, P_file, T_id, P_id, T_IPC, P_IPC).
# Simple/*.json records contain sequential file IDs, not actual EP publication numbers.
# Reconstruct this mapping separately from EPO_SR_parsed_260207/260207_Target_data.xlsx.
# This map is required to join records to XML drawings and SureChEMBL/PatCID by patent number.
ID_PATENT_MAP_PATH = os.path.join(PROJECT_DIR, 'data', 'reference', 'id_patent_map.csv')

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'samples')
BENCHMARK_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'benchmark', 'results')

# Legacy connected-record/time split from scripts/db_export/build_time_family_split.py.
# Not a family_id split or the final T123 five-fold assignment.
# Optional record filename -> 'train' | 'test' | 'train_unresolved' mapping.
# Without it, the pipeline still runs and computes only whole-dataset metrics.
TRAIN_TEST_SPLIT_PATH = os.path.join(DATA_DIR, 'reference', 'train_test_split.json')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
os.makedirs(BENCHMARK_OUTPUT_DIR, exist_ok=True)

# IPC Section C (chemistry/metallurgy: C01 ~ C99) and pharmaceutical subclasses (A61K, A61P).
SECTION_C_PREFIXES = ['C', 'A61K', 'A61P']

# Hybrid RAG weights, a single source matching the design formula:
# Score(Q,D) = alpha*TextSim + gamma*TanimotoSim + beta*VisionBonus; alpha+gamma+beta = 1.0.
# Weights when both structures (SMILES) and drawings are available. If either is missing,
# MultimodalChemicalRanker renormalizes available components (rag_engine/multimodal_ranker.py).
# Never silently substitute defaults or fabricate scores.
DEFAULT_TEXT_WEIGHT = 0.5
DEFAULT_TANIMOTO_WEIGHT = 0.3
DEFAULT_VISION_WEIGHT = 0.2

MAX_STREAM_SAMPLES = 2000

# ---- EPO OPS (Open Patent Services) API credentials ----
# The local XML corpus (EPO_XML_DIR) contains references such as <img file="imgf0001.tif"/>,
# but not actual drawing rasters (.tif); the original large archive is unavailable.
# Retrieve public drawings individually through EPO OPS as image inputs for
# OCSR (vision_ocsr/ocsr_pipeline.py). Set credentials via environment variables; never commit .env files.
EPO_OPS_CONSUMER_KEY = os.environ.get('EPO_OPS_CONSUMER_KEY')
EPO_OPS_CONSUMER_SECRET = os.environ.get('EPO_OPS_CONSUMER_SECRET')

# OCSR (drawing -> SMILES) switch, default False. Used only after SureChEMBL matching fails.
# This fallback downloads OPS drawings and recognizes structures with DECIMER;
# enable only after OPS quota and DECIMER are ready (see run_chem_rag.py).
ENABLE_OCSR_FALLBACK = os.environ.get('ENABLE_OCSR_FALLBACK', '0') == '1'

# Cache OPS pages, crops, and OCSR results so repeated patent lookups
# do not consume API quota again.
OCSR_CACHE_DIR = os.path.join(DATA_DIR, 'ocsr_cache')
os.makedirs(OCSR_CACHE_DIR, exist_ok=True)

# Maximum drawing pages per patent; protects the free-tier 3.5GB weekly traffic allowance.
OCSR_MAX_PAGES_PER_PATENT = 3
