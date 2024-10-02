#!/bin/bash

# This script performs inference on the Emotion6 dataset using four different metric projectors
# Each configuration consists of a specific vision encoder, a language model, 
# and a metric projector with pre-trained weights.

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

# OpenELM LLMs rely on LLaMA's tokenizer, so leave this as it is
# (we handle it further on at the LLM class)
TOKENIZER_PATH="meta-llama/Llama-2-7b-hf"



# TEST 1: METRIC PROJECTOR (CLIP VIT-L/14 + OpenELM-270M) = 0.70B
PROJECTOR_PATH="/Users/ggcr/Code/TFM/projector/TinyEmo-CLIP-OpenELM-270M-Syn.bin"
VISION_ENCODER_PATH="openai/clip-vit-large-patch14"
MODEL_PATH="apple/OpenELM-270M-Instruct"
LOSS_NAME="infoNCE"
echo $DATASET
VT_VARIANT="${VISION_ENCODER_PATH#*/}"
LLM_VARIANT="${MODEL_PATH#*/}"
python3 -m projector_mps.src.eval \
    --vision_encoder_path "$VISION_ENCODER_PATH" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --dataset "$DATASET" \
    --projector_weights_path "$PROJECTOR_PATH" \
    --per_device_eval_batch_size 8 \
    --loss_name "$LOSS_NAME" \
    --run_name test-${DATASET}-projector-${LLM_VARIANT}-${VT_VARIANT} \
    --precomputed "True" \


# TEST 2: METRIC PROJECTOR (CLIP VIT-L/14 + OpenELM-450M) = 0.88B
PROJECTOR_PATH="/Users/ggcr/Code/TFM/projector/TinyEmo-CLIP-OpenELM-450M.bin"
VISION_ENCODER_PATH="openai/clip-vit-large-patch14"
MODEL_PATH="apple/OpenELM-450M-Instruct"
LOSS_NAME="CosineEmbedding"
VT_VARIANT="${VISION_ENCODER_PATH#*/}"
LLM_VARIANT="${MODEL_PATH#*/}"
python3 -m projector_mps.src.eval \
    --vision_encoder_path "$VISION_ENCODER_PATH" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --dataset "$DATASET" \
    --projector_weights_path "$PROJECTOR_PATH" \
    --per_device_eval_batch_size 8 \
    --loss_name "$LOSS_NAME" \
    --run_name test-${DATASET}-projector-${LLM_VARIANT}-${VT_VARIANT} \
    --precomputed "True" \


# TEST 3: METRIC PROJECTOR (CLIP VIT-L/14 + TinyLlama-1.1) = 1.53B
PROJECTOR_PATH="/Users/ggcr/Code/TFM/projector/TinyEmo-CLIP-TinyLlama-1_1-Syn.bin"
VISION_ENCODER_PATH="openai/clip-vit-large-patch14"
MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOSS_NAME="CosineEmbedding"
VT_VARIANT="${VISION_ENCODER_PATH#*/}"
LLM_VARIANT="${MODEL_PATH#*/}"
python3 -m projector_mps.src.eval \
    --vision_encoder_path "$VISION_ENCODER_PATH" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --dataset "$DATASET" \
    --projector_weights_path "$PROJECTOR_PATH" \
    --per_device_eval_batch_size 8 \
    --loss_name "$LOSS_NAME" \
    --run_name test-${DATASET}-projector-${LLM_VARIANT}-${VT_VARIANT} \
    --precomputed "True" \


# TEST 4: METRIC PROJECTOR (CLIP VIT-L/14 + Phi-2) = 3.21B
PROJECTOR_PATH="/Users/ggcr/Code/TFM/projector/TinyEmo-CLIP-Phi-2.bin"
VISION_ENCODER_PATH="openai/clip-vit-large-patch14"
MODEL_PATH="microsoft/Phi-2"
LOSS_NAME="CosineEmbedding"
VT_VARIANT="${VISION_ENCODER_PATH#*/}"
LLM_VARIANT="${MODEL_PATH#*/}"
python3 -m projector_mps.src.eval \
    --vision_encoder_path "$VISION_ENCODER_PATH" \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --dataset "$DATASET" \
    --projector_weights_path "$PROJECTOR_PATH" \
    --per_device_eval_batch_size 8 \
    --loss_name "$LOSS_NAME" \
    --run_name test-${DATASET}-projector-${LLM_VARIANT}-${VT_VARIANT} \
    --precomputed "True" \