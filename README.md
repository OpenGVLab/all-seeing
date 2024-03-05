# The All-Seeing Project <img width="60" alt="image" src="https://github.com/OpenGVLab/all-seeing/assets/8529570/54c8d328-aa67-4d28-99de-90d019e8e7d0">

This is the official implementation of the following papers:

- [The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World](https://arxiv.org/abs/2308.01907)

- [The All-Seeing Project V2: Towards General Relation Comprehension of the Open World](https://arxiv.org/abs/2402.19474)

> The name "All-Seeing" is derived from "The All-Seeing Eye", which means having complete knowledge, awareness, or insight into all aspects of existence. The logo is Millennium Puzzle, an artifact from the manga "Yu-Gi-Oh!")

## News and Updates 🚀🚀🚀
- `Feb 28, 2024`: All-Seeing Project v2 is out! Our [**ASMv2**](https://huggingface.co/OpenGVLab/ASMv2) achieves state-of-the-art performance across a variety of image-level and region-level tasks! See [**here**](all-seeing-v2/README.md) for more details.
- `Feb 21, 2024`: [**ASM**](https://huggingface.co/OpenGVLab/ASM-FT), [**AS-Core**](https://huggingface.co/datasets/OpenGVLab/AS-Core), [**AS-10M**](https://huggingface.co/datasets/OpenGVLab/AS-V2/blob/main/as_pretrain_10m.json), [**AS-100M**](https://huggingface.co/datasets/OpenGVLab/AS-100M) is released!
- `Jan 16, 2024`: All-Seeing Project is accepted by ICLR 2024!
- `Aug 29, 2023`: [**All-Seeing Model Demo**](https://openxlab.org.cn/apps/detail/wangweiyun/All-Seeing-Model-Demo) and [**Dataset Browser**](https://openxlab.org.cn/apps/detail/wangweiyun/All-Seeing-Dataset-Browser) are available on the OpenXLab now!

## Schedule
- [x] Release the ASMv2 model.
- [x] Release the AS-V2 dataset.
- [x] Release the ASM model.
- [ ] Release the full version of AS-1B.
- [x] Release AS-Core, which is the human-verified subset of AS-1B.
- [x] Release AS-100M, which is the 100M subset of AS-1B.
- [x] Release AS-10M, which is the 10M subset of AS-1B.
- [x] Online demo, including dataset browser and ASM online demo.

<!-- ## Online Demo
**All-Seeing Model demo** is available [here](https://openxlab.org.cn/apps/detail/wangweiyun/All-Seeing-Model-Demo).

**Dataset Browser** is available [here](https://openxlab.org.cn/apps/detail/wangweiyun/All-Seeing-Dataset-Browser).



https://github.com/OpenGVLab/all-seeing/assets/47669167/9b5b32d1-863a-4579-b576-b82523f2205e -->

## Introduction 

<!-- ### [**The All-Seeing Project**](all-seeing/README.md) -->
### The All-Seeing Project [[Paper](https://arxiv.org/abs/2308.01907)][[Model](https://huggingface.co/OpenGVLab/ASM-FT)][[Dataset](https://huggingface.co/datasets/OpenGVLab/AS-100M)][[Code](all-seeing/README.md)]

[***All-Seeing 1B (AS-1B) dataset***](https://huggingface.co/datasets/OpenGVLab/AS-100M): we propose a new large-scale dataset (AS-1B) for open-world panoptic visual recognition and understanding, using an economical semi-automatic data engine that combines the power of off-the-shelf vision/language models and human feedback.

[***All-Seeing Model (ASM)***](https://huggingface.co/OpenGVLab/ASM-FT): we develop a unified vision-language foundation model (ASM) for open-world panoptic visual recognition and understanding. Aligning with LLMs, our ASM supports versatile image-text retrieval and generation tasks, demonstrating impressive zero-shot capability.

<!-- ### [**The All-Seeing Project V2**](all-seeing-v2/README.md) -->
### The All-Seeing Project V2 [[Paper](https://arxiv.org/abs/2402.19474)][[Model](https://huggingface.co/OpenGVLab/ASMv2)][[Dataset](https://huggingface.co/datasets/OpenGVLab/AS-V2)][[Code](all-seeing-v2/README.md)]

***[All-Seeing Dataset V2 (AS-V2) dataset](https://huggingface.co/datasets/OpenGVLab/AS-V2)***: we propose a novel task, termed Relation Conversation (ReC), which unifies the formulation of text generation, object localization, and relation comprehension. Based on the unified formulation, we construct the AS-V2 dataset, which consists of 127K high-quality relation conversation samples, to unlock the ReC capability for Multi-modal Large Language Models (MLLMs).

***[All-Seeing Model v2 (ASMv2)](https://huggingface.co/OpenGVLab/ASMv2)***: we develop ASMv2, which integrates the Relation Conversation ability while maintaining powerful general capabilities.
It is endowed with grounding and referring capabilities, exhibiting state-of-the-art performance on region-level tasks.
Furthermore, this model can be naturally adapted to the Scene Graph Generation task in an open-ended manner.

***[Circular-based Relation Probing Evaluation (CRPE) benchmark](https://huggingface.co/datasets/OpenGVLab/CRPE)***: We construct a benchmark called Circular-based Relation Probing Evaluation (CRPE), which is the first benchmark that covers all elements of the relation triplets `(subject, predicate, object)`, providing a systematic platform for the evaluation of relation comprehension ability.

## License

This project is released under the [Apache 2.0 license](LICENSE). 


## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{wang2023allseeing,
  title={The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World},
  author={Wang, Weiyun and Shi, Min and Li, Qingyun and Wang, Wenhai and Huang, Zhenhang and Xing, Linjie and Chen, Zhe and Li, Hao and Zhu, Xizhou and Cao, Zhiguo and others},
  journal={arXiv preprint arXiv:2308.01907},
  year={2023}
}
@article{wang2024allseeing_v2,
  title={The All-Seeing Project V2: Towards General Relation Comprehension of the Open World},
  author={Wang, Weiyun and Ren, Yiming and Luo, Haowen and Li, Tiantong and Yan, Chenxiang and Chen, Zhe and Wang, Wenhai and Li, Qingyun and Lu, Lewei and Zhu, Xizhou and others},
  journal={arXiv preprint arXiv:2402.19474},
  year={2024}
}
```
