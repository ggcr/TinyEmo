#!/bin/bash

# ============= VISION ENCODERS AVAILABLE =============
VISION_ENCODER_PATH="openai/clip-vit-large-patch14"
# VISION_ENCODER_PATH="google/siglip-so400m-patch14-384"
# VISION_ENCODER_PATH="facebook/dinov2-large"
# VISION_ENCODER_PATH="openai/clip-vit-base-patch16"
# =====================================================

# ================== LLMs AVAILABLE ===================
MODEL_PATH="apple/OpenELM-270M-Instruct"
# MODEL_PATH="apple/OpenELM-450M-Instruct"
# MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_PATH="microsoft/Phi-2"
# MODEL_PATH="microsoft/Phi-3-mini-128k-instruct"
# =====================================================

TOKENIZER_PATH="meta-llama/Llama-2-7b-hf"

VT_VARIANT="${VISION_ENCODER_PATH#*/}"
LLM_VARIANT="${MODEL_PATH#*/}"

DATASET="FI" # ["WEBEmo", "EmoSet", "Emotion6", "EmoROI", "UnbiasedEmo"]
python3 -m src.model.precompute_features \
    --vision_encoder_path "$VISION_ENCODER_PATH" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --dataset "$DATASET" \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 2 \
    --report_to "wandb" \
    --run_name "precompute-features-${VT_VARIANT}-${DATASET}" \
    --output_dir "/home/cgutierrez/checkpoints/projector-${VT_VARIANT}-FI/"
