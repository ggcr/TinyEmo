#!/bin/bash

# This script performs inference on the Emotion6 dataset using four different metric projectors
# Each configuration consists of a specific projector path
# We already provide the precomputed Emotion6 CLIP features `TinyEmo/projector_mps/sample_data/`

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

# Note: Each dataset requires a custom dataloader due to structural differences.
# We provide examples in `TinyEmo/projector_mps/dataloaders` as starting points.
# Precomputed dataloaders (with CLIP embeddings) are available to reduce computation,
# and we encourage using this approach for efficiency.


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


# TEST 1: METRIC PROJECTOR (CLIP VIT-L/14 + OpenELM-270M) = 0.70B
PROJECTOR_MODEL="ggcristian/TinyEmo-CLIP-OpenELM-270M-Syn"
python3 -m projector_mps.src.eval \
    --projector_path "$PROJECTOR_MODEL" \
    --dataset "$DATASET" \
    --per_device_eval_batch_size 8 \
    --run_name test-${DATASET}-projector-OpenELM-270M-CLIP \
    --precomputed "True"

# TEST 2: METRIC PROJECTOR (CLIP VIT-L/14 + OpenELM-450M) = 0.88B
PROJECTOR_MODEL="ggcristian/TinyEmo-CLIP-OpenELM-450M"
python3 -m projector_mps.src.eval \
    --projector_path "$PROJECTOR_MODEL" \
    --dataset "$DATASET" \
    --per_device_eval_batch_size 8 \
    --run_name test-${DATASET}-projector-OpenELM-450M-CLIP \
    --precomputed "True"

# TEST 3: METRIC PROJECTOR (CLIP VIT-L/14 + TinyLlama-1.1) = 1.53B
PROJECTOR_MODEL="ggcristian/TinyEmo-CLIP-TinyLlama-1_1-Syn"
python3 -m projector_mps.src.eval \
    --projector_path "$PROJECTOR_MODEL" \
    --dataset "$DATASET" \
    --per_device_eval_batch_size 8 \
    --run_name test-${DATASET}-projector-TinyLlama-1_1-CLIP \
    --precomputed "True"

# TEST 4: METRIC PROJECTOR (CLIP VIT-L/14 + Phi-2) = 3.21B
PROJECTOR_MODEL="ggcristian/TinyEmo-CLIP-Phi-2"
python3 -m projector_mps.src.eval \
    --projector_path "$PROJECTOR_MODEL" \
    --dataset "$DATASET" \
    --per_device_eval_batch_size 8 \
    --run_name test-${DATASET}-projector-Phi-2-CLIP \
    --precomputed "True"
