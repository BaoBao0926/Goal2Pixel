set -x

DATASET=${1}
DATA=${2}



if [ "${DATASET}" == "r2r" ]; then
  if [ "${IS_ROTATE_MODE}" == "true" ]; then
    EXP_NAME="${EXP_NAME:-write_mp3d_vln_v1_r2r_valunseen}"
  else
    EXP_NAME="${EXP_NAME:-write_mp3d_vln_v1_r2r}"
  fi
fi
if [ "${DATASET}" == "rxr-en-US" ]; then
  if [ "${IS_ROTATE_MODE}" == "true" ]; then
    EXP_NAME="${EXP_NAME:-write_mp3d_vln_v1_rxr-en-US_valunseen}"
  else
    EXP_NAME="${EXP_NAME:-write_mp3d_vln_v1_rxr-en-US}"
  fi
fi
if [ "${DATASET}" == "rxr-en-IN" ]; then
  if [ "${IS_ROTATE_MODE}" == "true" ]; then
    EXP_NAME="${EXP_NAME:-write_mp3d_vln_v1_rxr-en-IN_valunseen}"
  else
    EXP_NAME="${EXP_NAME:-write_mp3d_vln_v1_rxr-en-IN}"
  fi
fi



# Set GPU to use
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# wandb if needed
WANDB_DIR=$(pwd)/all_log
export WANDB_DIR

# Set PYTHONPATH to include root, scripts, habitat-baselines, and habitat-lab
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/scripts:$(pwd)/habitat-lab/habitat-baselines:$(pwd)/habitat-lab/habitat-lab:$(pwd)/Grounded-SAM-2:$(pwd)/Grounded-SAM-2/grounding_dino

# Move into habitat-lab directory
cd habitat-lab/ || exit 1

# Quiet debug logs
export HABITAT_ENV_DEBUG=0
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# # Launch training

# R2R
if [ ${DATA} == "trajectory" ]; then
  python -u -m scripts.scripts_for_generating_data.write_mp3d_data_trajectory \
    --config-name="${EXP_NAME}.yaml" \
    --config-path="../../configs_hydra/" 
  exit 0
fi
if [ ${DATA} == "pixel" ]; then
  python -u -m scripts.scripts_for_generating_data.write_mp3d_data_pixel \
    --config-name="${EXP_NAME}.yaml" \
    --config-path="../../configs_hydra/" 
  exit 0
fi
if [ ${DATA} == "history" ]; then
  python -u -m scripts.scripts_for_generating_data.write_mp3d_data_history \
    --config-name="${EXP_NAME}.yaml" \
    --config-path="../../configs_hydra/" \
    --hist-alg="og"
  exit 0
fi
if [ ${DATA} == "history-SIFT" ]; then
  python -u -m scripts.scripts_for_generating_data.write_mp3d_data_history \
    --hist-alg="cv" \
    --config-name="${EXP_NAME}.yaml" \
    --config-path="../../configs_hydra/" 
  exit 0
fi



