#!/bin/bash

# ============= VISION ENCODERS AVAILABLE =============
VISION_ENCODER_PATH="openai/clip-vit-large-patch14"
# VISION_ENCODER_PATH="google/siglip-so400m-patch14-384"
# VISION_ENCODER_PATH="facebook/dinov2-large"
# VISION_ENCODER_PATH="openai/clip-vit-base-patch16"
# =====================================================

# ================== LLMs AVAILABLE ===================
MODEL_PATH="apple/OpenELM-450M-Instruct"
# MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_PATH="apple/OpenELM-270M-Instruct"
# MODEL_PATH="microsoft/Phi-2"
# MODEL_PATH="microsoft/Phi-3-mini-128k-instruct"
# =====================================================

# ================== AVAILABLE LOSSES ===================
LOSS_NAME="CosineEmbedding"
# LOSS_NAME="NTXent"
# LOSS_NAME="infoNCE"
# =======================================================

# OpenELM LLMs rely on LLaMA's tokenizer, so we will set this set
TOKENIZER_PATH="meta-llama/Llama-2-7b-hf"

CONNECTOR="mlp"
DATASET="Joint" # Joint includes WEBEmo, EmoSet and EmotionROI

VT_VARIANT="${VISION_ENCODER_PATH#*/}"
LLM_VARIANT="${MODEL_PATH#*/}"

python3 -m src.train_aug \
    --vision_encoder_path "$VISION_ENCODER_PATH" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --dataset "$DATASET" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 5 \
    --report_to "wandb" \
    --run_name joint_pretraining-${LLM_VARIANT}-${VT_VARIANT}-${DATASET} \
    --output_dir /home/cgutierrez/shared/checkpoints/projector-${LLM_VARIANT}-${VT_VARIANT}-${CONNECTOR}-${DATASET}_Augmented/ \
    --precomputed "True" \
