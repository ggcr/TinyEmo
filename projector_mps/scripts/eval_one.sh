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

# LOSS_NAME="NTXent"
LOSS_NAME="CosineEmbedding"
# LOSS_NAME="infoNCE"

PROJECTOR_PATH="/home/cgutierrez/shared/checkpoints/projector-Phi-2-clip-vit-large-patch14-EmoROI_EmoROI/pytorch_model_5.bin"
VT_VARIANT="${VISION_ENCODER_PATH#*/}"
LLM_VARIANT="${MODEL_PATH#*/}"

python3 -m src.eval_one \
    --vision_encoder_path "$VISION_ENCODER_PATH" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --projector_weights_path "$PROJECTOR_PATH" \
    --loss_name "$LOSS_NAME" \
