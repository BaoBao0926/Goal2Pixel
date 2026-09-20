set -euo pipefail   # exit on error, undefined vars, or pipefail
set -x

# in configs_vlm folder
EXP_NAME=${EXP_NAME:-"sample_debug_99779"}
PROJECT_NAME=${PROJECT_NAME:-"COCO-LoRA-InternVL"} # Set the default project name here

# GPUs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # e.g. "0,1,2,3"
if [[ -z "${GPUS:-}" ]]; then
  IFS=, read -ra _g <<< "$CUDA_VISIBLE_DEVICES"
  GPUS=${#_g[@]}
fi
: "${GPUS:=1}"

# batches
: "${GRADIENT_ACC:=4}"           # total number of images processed at each forward pass across all GPUs
: "${PER_DEVICE_BATCH_SIZE:=1}"  # per GPU
: "${MAX_STEPS:=2000}"        # total number of training steps
: "${META_FILE_NAME:='debug.json'}"
# ensure integer >= 1
BATCH_SIZE=$(( PER_DEVICE_BATCH_SIZE * GPUS * GRADIENT_ACC ))

echo "Project Name: $PROJECT_NAME;
Experiment Name: $EXP_NAME;
Total batch size: $BATCH_SIZE;
Per device: $PER_DEVICE_BATCH_SIZE;
GPU: $GPUS;
Gradient Accumulation: $GRADIENT_ACC;
Max steps: $MAX_STEPS;"

# -------- batch sizes --------
export PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE"
export BATCH_SIZE="$BATCH_SIZE"
export GRADIENT_ACC="$GRADIENT_ACC"
export META_FILE_NAME="$META_FILE_NAME"
export MAX_STEPS="$MAX_STEPS"

# -------- envs --------
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/InternVL_cleaned/internvl_chat"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}"
# TensorFlow log level to suppress most of the logs.
# 3 only allows error messages to be printed
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export USE_TCS_LOADER=0
export PYTHONWARNINGS="ignore::FutureWarning"

# optional: helps CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# -------- wandb --------
BASE_META_NAME="${META_FILE_NAME%.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/all_log/my_experiments/${EXP_NAME}_${BASE_META_NAME}}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


export OUTPUT_DIR=$OUTPUT_DIR
export JOBLOG=${OUTPUT_DIR}/training.log

export WANDB_DIR=${OUTPUT_DIR}
export WANDB_PROJECT=$PROJECT_NAME
export WANDB_ENTITY=${WANDB_ENTITY:-"tsaisplus-nanyang-technological-university-singapore"}
export WANDB_NAME="${WANDB_NAME:-${EXP_NAME}_steps${MAX_STEPS}_gpus${GPUS}_acc${GRADIENT_ACC}}"
export WANDB_LOG_MODEL=false
export WANDB_WATCH=false
export WANDB_DISABLED=false
export DEEPSPEED_LOG_LEVEL=warning

# -------- go to code --------
cd InternVL_cleaned/internvl_chat || { echo "cd failed"; exit 1; }

CFG="../../configs_vlm/${EXP_NAME}.json"
if [[ ! -f "$CFG" ]]; then
  echo "❌ Config not found: $CFG" >&2
  exit 1
fi


torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --nproc_per_node="$GPUS" \
  --master_port="$MASTER_PORT" \
  internvl_cleaned/train/internvl_chat_finetune.py \
  "$CFG" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"