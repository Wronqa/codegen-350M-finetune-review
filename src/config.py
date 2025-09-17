from pathlib import Path
import logging

PROJ_ROOT = Path(__file__).resolve().parents[1]  
logger = logging.getLogger(__name__)
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

CONFIGS_DIR = PROJ_ROOT / "configs"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREPARED_DATA_DIR = DATA_DIR / "prepared"
OUTPUTS_DIR = PROJ_ROOT / "outputs"
ADAPTERS_DIR = PROJ_ROOT / "adapters" /"codegen350m_lora"

DEFAULT_DATA_CONFIG = CONFIGS_DIR / "data.yaml"

TRAIN_PATH = PREPARED_DATA_DIR / "train.jsonl"
VAL_PATH   = PREPARED_DATA_DIR / "val.jsonl"

MODEL_ID  = "Salesforce/codegen-350M-multi"

SEED = 42
MAX_EXAMPLES_PER_SOURCE = 70_000

DEFAULT_TRAIN_RATIO = 0.9
DEFAULT_MAX_EXAMPLES_FINAL = 50_000
DEFAULT_MAX_RESPONSE_CHARS = 300

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

SEED = 42
MAX_SEQ_LEN = 512
PER_DEV_BATCH = 1
GRAD_ACCUM = 8
LR = 2e-4
EPOCHS = 1
FP16 = True
WARMUP_RATIO = 0.03

LOGGING_STEPS = 25
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
LR_SCHEDULER = "cosine"
PACKING = False
PADDING_SIDE = "right"
TRUNCATION_SIDE = "left"

MLFLOW_TRAIN_EXPERIMENT_NAME = "codegen-review"
MLFLOW_EVAL_EXPERIMENT_NAME = "eval_codegen"

EVAL_REPORT_DIR    = OUTPUTS_DIR         
EVAL_MAX_NEW_TOKENS = 64
EVAL_N_SAMPLES      = None               
BERT_LANG           = "en"   

EVAL_LOG_EVERY: int = 50          
EVAL_GEN_TIMEOUT_S: float = 15.0 