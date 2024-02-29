# The All-Seeing Project <img width="60" alt="image" src="https://github.com/OpenGVLab/all-seeing/assets/8529570/54c8d328-aa67-4d28-99de-90d019e8e7d0"> [[Paper](https://arxiv.org/abs/2308.01907)][[Model](https://huggingface.co/OpenGVLab/ASM-FT)][[Dataset](https://huggingface.co/datasets/OpenGVLab/AS-100M)]

## Introduction 
We present the All-Seeing Project with:

[***All-Seeing 1B (AS-1B) dataset***](https://huggingface.co/datasets/OpenGVLab/AS-100M): we propose a new large-scale dataset (AS-1B) for open-world panoptic visual recognition and understanding, using an economical semi-automatic data engine that combines the power of off-the-shelf vision/language models and human feedback.

[***All-Seeing Model (ASM)***](https://huggingface.co/OpenGVLab/ASM-FT): we develop a unified vision-language foundation model (ASM) for open-world panoptic visual recognition and understanding. Aligning with LLMs, our ASM supports versatile image-text retrieval and generation tasks, demonstrating impressive zero-shot capability.

<img width="820" alt="image" src="https://github.com/OpenGVLab/all-seeing/assets/8529570/e43ab8db-6437-46f1-8aa1-c95f012e9147">


Figure 1: Overview and comparison of our All-Seeing project with other popular large foundation models.


## Dataset Overview
AS-1B with over 1 billion regions annotated with semantic tags, question-answering pairs, and detailed captions. It covers a wide range of 3.5 million common and rare concepts in the real world, and has 132.2 billion tokens that describe the concepts and their attributes.

<img width="800" alt="image" src="https://github.com/OpenGVLab/all-seeing/assets/8529570/adac37ed-312f-4f11-ba8a-6bc62067438f">


Some examples

<img width="800" alt="image" src="https://github.com/OpenGVLab/all-seeing/assets/8529570/fcf6ab07-c4ba-441c-aa6c-111c769f75b1">


## Model Architecture

The All-Seeing model (ASM) is a unified framework for panoptic visual recognition and understanding, including image/region-text retrieval, image/region recognition, captioning, and question-answering.
<img width="820" alt="image" src="https://github.com/OpenGVLab/all-seeing/assets/8529570/8995e88c-6381-452f-91e4-05d68a2795fc">


## Installation

```
pip install torch==2.0.0
pip install transformers==4.28.0
pip install pycocoevalcap
pip install mmeval==0.2.1
```

## Model Zoo

| Model                   | Download                                                               | Note                             |
| ----------------------- | ---------------------------------------------------------------------- | -------------------------------- |
| All-Seeing-Model-Pretrain      | 🤗 [HF link](https://huggingface.co/OpenGVLab/ASM-Pretrain)      | a unified vision-language foundation model          |
| All-Seeing-Model-FT      | 🤗 [HF link](https://huggingface.co/OpenGVLab/ASM-FT)      | a vision-language foundation model for region-level qa          |

## Testing

For evaluation of region captioning, please download the [evaluation data annotations](https://huggingface.co/datasets/OpenGVLab/Caption-Evaluation-Data/tree/main) and put them in `./data` as the following structure.

The images can be downloaded from:

- [COCO](https://cocodataset.org/)
- [LVIS](https://www.lvisdataset.org/)
- [VG](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)
- [Flickr30K](https://hockenmaier.cs.illinois.edu/DenotationGraph/)
- [NoCaps](https://nocaps.org/)

```
├── coco
│   ├── val2014
│   ├── val2017
│   ├── train2017
│   └── annotations
├── lvis
│   ├── val2017
│   └── annotations
├── flickr30k
│   └── images
├── nocaps
│   └── val
├── vg
│   ├── VG_100K
│   └── VG_100K_2
├── coco_karpathy_val.json
├── flickr30k_karpathy_test.json
├── nocaps_val.json
├── refcocog_test_coco_format.json
├── refcocog_val_coco_format.json
└── vg_test_coco_format.json
```

- Image/Region Captioning

```shell
# supported dataset: coco_caption, flickr30k, nocaps, vg, refcocog_val, refcocog_test
sh scripts/eval_caption.sh OpenGVLab/All-Seeing-Model-Pretrain ${DATASET_NAME}
```

- Region Recognition

```shell
# supported dataset: coco, lvis
sh scripts/eval_region_recognition.sh OpenGVLab/All-Seeing-Model-Pretrain ${DATASET_NAME}
```
