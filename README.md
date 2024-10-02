# TinyEmo

[Paper]

[[Metric Projector Card]](https://huggingface.co/collections/ggcristian/tinyemo-projectors-66fd14187fbd5d30764abc24) [TinyEmo MM-LLM Card]

[[EmoReason Dataset card]](https://huggingface.co/collections/ggcristian/tinyemo-emoreason-dataset-66fd16963c945fb7058a8f55)

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

### Metric Projector inference

We provide precomputed CLIP features for the Emotion6 dataset, and you can evaluate them using two methods:

#### Our Projectors from Hugging Face

To evaluate the projectors from Hugging Face, use the [scripts/eval.sh](https://github.com/ggcr/TinyEmo/blob/main/projector_mps/scripts/eval.sh) script:

```bash
conda activate projector_mps
bash projector_mps/scripts/eval.sh
```

Below is a table of the available projectors:

| Model Architecture                     | Parameters | Zero-shot Accuracy | HuggingFace Link                                                                 |
|----------------------------------------| ---------- |--------------------|----------------------------------------------------------------------|
| CLIP ViT-L/14 + OpenELM-270M-I         | 0.70B      | 57.87%             | [HF Projector 0.70B Link](https://huggingface.co/ggcristian/TinyEmo-CLIP-OpenELM-270M) |
| CLIP ViT-L/14 + OpenELM-450M-I         | 0.88B      | 55.24%             | [HF Projector 0.88B Link](https://huggingface.co/ggcristian/TinyEmo-CLIP-OpenELM-450M) |
| CLIP ViT-L/14 + TinyLLaMA 1.1          | 1.53B      | 56.13%             | [HF Projector 1.53B Link](https://huggingface.co/ggcristian/TinyEmo-CLIP-TinyLlama-1_1-Syn) |
| CLIP ViT-L/14 + Microsoft Phi 2        | 3.21B      | 56.28%             | [HF Projector 3.21B Link](https://huggingface.co/ggcristian/TinyEmo-CLIP-Phi-2)      |

#### Custom Projectors with Local Weights

To use custom local weights or models, run the following:

```bash
conda activate projector_mps
bash projector_mps/scripts/eval_custom.sh
```

This allows you to specify different vision encoders, language models, and loss functions, as well as use your own projector weights.


## Acknowledgement

The Metric Projector was built from the foundations of [CLIP-E](https://arxiv.org/abs/2310.12062) paper!

Our codebase for the MM-LLM is forked from the [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory) project.

## Citation

```
@mastersthesis{gutierrez2024tinyemo,
  title        = {TinyEmo: Scaling down Emotional Reasoning via Metric Projection},
  author       = {Cristian Gutierrez},
  year         = 2024,
  month        = {September},
  address      = {Barcelona, Spain},
  school       = {Universitat Autonoma de Barcelona (UAB)},
  type         = {Master's thesis}
}
```