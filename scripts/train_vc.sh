#!/usr/bin/env bash
set -euo pipefail

# Submit F5-TTS training with vc. Run from anywhere.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/log}"
mkdir -p "${LOG_DIR}"

# ---- vc submit settings (override via env) ----
VC_IMAGE="${VC_IMAGE:-docker.v2.aispeech.com/hpc-base/ai_base-pytorch-asr:pytorch_1.10-cuda_11.1-0.0.1-004}"
VC_PARTITION="${VC_PARTITION:-pdgpu-a10}"
VC_GPU_PER_TASK="${VC_GPU_PER_TASK:-1}"
VC_MEM_PER_TASK="${VC_MEM_PER_TASK:-8G}"
VC_CPU_PER_TASK="${VC_CPU_PER_TASK:-3}"
VC_JOB_NAME="${VC_JOB_NAME:-f5tts_train}"
VC_TASK_RANGE="${VC_TASK_RANGE:-JOB=1:1}"

# ---- 训练设置 ----
# 单配置模式：设置 CONFIG_NAME 即可（覆盖 CONFIG_LIST）
CONFIG_NAME="${CONFIG_NAME:-}"
# 多配置模式：默认跑 Small + Tiny1~Tiny4，可用空格分隔自定义列表
CONFIG_LIST="${CONFIG_LIST:-F5TTS_v1_Small.yaml F5TTS_v1_Tiny1.yaml F5TTS_v1_Tiny2.yaml F5TTS_v1_Tiny3.yaml F5TTS_v1_Tiny4.yaml}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
# 可选：覆盖 datasets.name
DATASET_NAME="${DATASET_NAME:-}"
# 可选：传入 hydra 覆盖参数
EXTRA_ARGS="${EXTRA_ARGS:-}"  # 例：++optim.epochs=1 ++datasets.batch_size_per_gpu=4000
# 若 REQUIRE_WANDB=1 则强制要求 WANDB_API_KEY
REQUIRE_WANDB="${REQUIRE_WANDB:-0}"

# ---- env passthrough (optional) ----
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# 组装配置列表
if [[ -n "${CONFIG_NAME}" ]]; then
  CONFIGS=( "${CONFIG_NAME}" )
else
  # shellcheck disable=SC2206
  CONFIGS=( ${CONFIG_LIST} )
fi

if [[ "${REQUIRE_WANDB}" == "1" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: REQUIRE_WANDB=1 但未设置 WANDB_API_KEY"
  exit 1
fi

# Local test mode (no vc submit): LOCAL_RUN=1 ./scripts/train_vc.sh
if [[ "${LOCAL_RUN:-0}" == "1" ]]; then
  for cfg in "${CONFIGS[@]}"; do
    ACCELERATE_CMD=(
      accelerate launch
      --mixed_precision="${MIXED_PRECISION}"
      "${ROOT_DIR}/src/f5_tts/train/train.py"
      --config-name "${cfg}"
    )
    if [[ -n "${DATASET_NAME}" ]]; then
      ACCELERATE_CMD+=( "++datasets.name=${DATASET_NAME}" )
    fi
    if [[ -n "${EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      ACCELERATE_CMD+=( ${EXTRA_ARGS} )
    fi
    echo "[LOCAL RUN] ${ACCELERATE_CMD[*]}"
    "${ACCELERATE_CMD[@]}"
  done
  exit 0
fi

# Build vc submit command
VC_CMD=(
  vc submit
  --image "${VC_IMAGE}"
  --partition "${VC_PARTITION}"
  --gpu-per-task "${VC_GPU_PER_TASK}"
  --mem-per-task "${VC_MEM_PER_TASK}"
  --cpu-per-task "${VC_CPU_PER_TASK}"
)

for cfg in "${CONFIGS[@]}"; do
  cfg_base="${cfg%.yaml}"
  job_name="${VC_JOB_NAME}_${cfg_base}"
  log_path="${LOG_DIR}/train_${cfg_base}.JOB.log"
  ACCELERATE_CMD=(
    accelerate launch
    --mixed_precision="${MIXED_PRECISION}"
    "${ROOT_DIR}/src/f5_tts/train/train.py"
    --config-name "${cfg}"
  )
  if [[ -n "${DATASET_NAME}" ]]; then
    ACCELERATE_CMD+=( "++datasets.name=${DATASET_NAME}" )
  fi
  if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    ACCELERATE_CMD+=( ${EXTRA_ARGS} )
  fi

  echo "[VC SUBMIT] ${VC_CMD[*]} --sync --job ${job_name} ${VC_TASK_RANGE} ${log_path}"
  echo "[CMD] ${ACCELERATE_CMD[*]}"

  "${VC_CMD[@]}" --sync --job "${job_name}" "${VC_TASK_RANGE}" \
    "${log_path}" --cmd "${ACCELERATE_CMD[*]}"
done

