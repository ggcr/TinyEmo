# TinyEmo

[Paper]

[Metric Projector Card] [TinyEmo MM-LLM Card]

[[Reasoning Pre-training Dataset]](https://huggingface.co/datasets/ggcristian/TinyEmo-Pretrain-525k) [[Reasoning Fine-tuning Dataset]](https://huggingface.co/datasets/ggcristian/TinyEmo-EmoReason-175k) [[Reasoning Claude Dataset]](https://huggingface.co/datasets/ggcristian/TinyEmo-EmoReasonHQ-Claude-1.4k)

TinyEmo is a family of small multi-modal language models for emotional reasoning and classification. Our
approach features: (1) a synthetic emotional instruct dataset for both pre-training and fine-tuning stages, (2) a Metric Projector
that delegates classification from the language model allowing for more efficient training and inference, (3) a multi-modal large
language model (MM-LLM) for emotional reasoning, and (4) a semi-automated framework for bias detection. TinyEmo is able to
perform emotion classification and emotional reasoning, all while using substantially fewer parameters than comparable models.
This efficiency allows us to freely incorporate more diverse emotional datasets, enabling strong performance on classification tasks,
with our smallest model (700M parameters) outperforming larger state-of-the-art models based on general-purpose MM-LLMs
with over 7B parameters. Additionally, the Metric Projector allows for interpretability and indirect bias detection in large models
without additional training, offering an approach to understand and improve AI systems.

## Installation and Requirements

### Metric Projector (Classification)

1. Clone this repository and navigate to the root of the project:
```
git clone https://github.com/ggcr/TinyEmo.git
cd TinyEmo
```

2. Create an environment and install dependencies:
```
conda create -n projector_mps python=3.10 -y
conda activate projector_mps
pip install --upgrade pip  # enable PEP 660 support
pip install -e projector_mps/.
```

### MM-LLM (Reasoning)

Refer to the [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory) installation section.

## Quickstart

### Metric Projector inference on Emotion6 dataset

In the scripts section we provide the data and the script to evaluate on Emotion6 (see `TinyEmo/scripts/eval_Emotion6.sh`).

```
conda activate projector_mps
bash projector_mps/scripts/eval_Emotion6.sh
```

## Acknowledgement

Our codebase for the MM-LLM is forked from the [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory) project.

## Citation

```
@mastersthesis{gutierrez2024tinyemo,
  title        = {TinyEmo: Scaling down Emotional Reasoning via Metric Projection},
  author       = {Cristian Gutierrez},
  year         = 2024,
  month        = {September},
  address      = {Barcelona, Spain},
  school       = {Universitat Autònoma de Barcelona (UAB)},
  type         = {Master's thesis}
}
```