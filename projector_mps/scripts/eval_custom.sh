#!/bin/bash

# ================ AVAILABLE DATASETS ================
DATASET="Emotion6" # we provide a precomputed Emotion6 dataset at `TinyEmo/projector_mps/sample_data/`
# DATASET="EmoSet"
# DATASET="WEBEmo"
# DATASET="EmoROI"
# DATASET="FI"
# DATASET="ArtPhoto"
# DATASET="Abstract"
# DATASET="UnbiasedEmo"
# ====================================================

# ============= VISION ENCODERS AVAILABLE =============
# VISION_ENCODER_PATH="openai/clip-vit-large-patch14"
# VISION_ENCODER_PATH="google/siglip-so400m-patch14-384"
# VISION_ENCODER_PATH="facebook/dinov2-large"
# VISION_ENCODER_PATH="openai/clip-vit-base-patch16"
# =====================================================

# ================== LLMs AVAILABLE ===================
# MODEL_PATH="apple/OpenELM-270M-Instruct"
# MODEL_PATH="apple/OpenELM-450M-Instruct"
# MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_PATH="microsoft/Phi-2"
# MODEL_PATH="microsoft/Phi-3-mini-128k-instruct"
# =====================================================

# ================== AVAILABLE LOSSES ===================
# LOSS_NAME="CosineEmbedding"
# LOSS_NAME="NTXent"
# LOSS_NAME="infoNCE"
# =======================================================



# CUSTOM MODEL
# You can specify the parts of your local model and a local path to the weights
VISION_ENCODER_PATH="openai/clip-vit-large-patch14"
MODEL_PATH="apple/OpenELM-270M-Instruct"
LOSS_NAME="infoNCE"
PROJECTOR_WEIGHTS_LOCAL="/Users/ggcr/Code/TFM/projector/TinyEmo-CLIP-OpenELM-270M-Syn.bin"
python3 -m projector_mps.src.eval \
    --vision_encoder_path "$VISION_ENCODER_PATH" \
    --model_path "$MODEL_PATH" \
    --projector_path "$PROJECTOR_WEIGHTS_LOCAL" \
    --dataset "$DATASET" \
    --per_device_eval_batch_size 8 \
    --loss_name "$LOSS_NAME" \
    --run_name test-${DATASET}-projector-OpenELM-270M-CLIP \
    --precomputed "True"
