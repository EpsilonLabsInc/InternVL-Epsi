#!/usr/bin/env bash
set -x

###############################################################################
# Configuration for multi-node training
###############################################################################
# The total number of nodes in the cluster
export NNODES=${NNODES:-2}

# The rank of this node (0 to NNODES-1)
export NODE_RANK=${NODE_RANK:-0}

# The number of GPUs on each node
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Address (IP or hostname) of the node with rank=0
export MASTER_ADDR=${MASTER_ADDR:-"node0"}

# Free port for communication
export MASTER_PORT=${MASTER_PORT:-29500}

###############################################################################
# Other training parameters (same as your single-node script)
###############################################################################
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))  # or adjust as needed

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LR=1e-5
prefix="/mnt/gradient_batch123/training/"
this_run="internvl2.5_26b_finetune_lora_${TIMESTAMP}_${LR}_all_data"
OUTPUT_DIR="${prefix}${this_run}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

###############################################################################
# Run the training with torchrun for multi-node
# IMPORTANT: --nnodes, --node_rank, --master_addr, --master_port must match
# or be set by environment variables on each node.
###############################################################################
torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "./pretrained/InternVL2_5-26B" \
  --conv_style "internlm2-chat" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "./shell/data/gradient_mimic_chexpert.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 48 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 50 \
  --learning_rate ${LR} \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config.json" \
  --max_grad_norm 1.0 \
  --report_to "wandb" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
