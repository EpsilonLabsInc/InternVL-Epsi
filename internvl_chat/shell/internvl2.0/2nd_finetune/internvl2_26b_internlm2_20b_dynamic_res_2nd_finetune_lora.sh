set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LR=1e-5

prefix="/home/eric/projects/InternVL-Epsi/internvl_chat/training/"

# OUTPUT_DIR="/mnt/data/ruian/internvl2/internvl2_26b_internlm2_20b_dynamic_res_2nd_finetune_lora_${TIMESTAMP}_${LR}"

this_run="internvl2.5_26b_finetune_lora_${TIMESTAMP}_${LR}_1labels-vlm2class"
this_run="internvl2.5_26b_finetune_lora_${TIMESTAMP}_${LR}_1labels-3k"
this_run="internvl2.5_26b_finetune_lora_${TIMESTAMP}_${LR}_1labels-10k"
this_run="internvl2.5_26b_finetune_lora_${TIMESTAMP}_${LR}_1labels-10k-mimic-chex"

this_run="26b_${TIMESTAMP}_${LR}_2.5_mimic2_${MAX_DYNAMIC_PATCH}_no_labels"
OUTPUT_DIR="${prefix}${this_run}"


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 2
# batch size per gpu: 2
# gradient accumulation steps: 4
# total batch size: 16
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "./pretrained/InternVL2_5-26B" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/mimic2_0320_nolabels.json" \
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
  --wandb_project "internvl2.5_26b_finetune_lora_mimic2" \
  --wandb_run_name "${TIMESTAMP}_nolabel" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
