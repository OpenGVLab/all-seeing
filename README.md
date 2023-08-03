# All-Seeing Project <img width="60" alt="image" src="https://github.com/Weiyun1025/all-seeing/assets/8529570/926c1393-ee84-41af-a329-7c2ecd236e18"> [[Paper](TODO)]

This is the official implementation of the paper "The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World". (The name "All-Seeing" is derived from "The All-Seeing Eye", which means having complete knowledge, awareness, or insight into all aspects of existence. The logo is Millennium Puzzle, an artifact from the manga "Yu-Gi-Oh!")

## Introduction 
We present the All-Seeing Project with:

***All-Seeing 1B (AS-1B) dataset***: we propose a new large-scale dataset (AS-1B) for open-world panoptic visual recognition and understanding, using an economical semi-automatic data engine that combines the power of off-the-shelf vision/language models and human feedback.

***All-Seeing Model (ASM)***: we develop a unified vision-language foundation model (ASM) for open-world panoptic visual recognition and understanding. Aligning with LLMs, our ASM supports versatile image-text retrieval and generation tasks, demonstrating impressive zero-shot capability.

<img width="820" alt="image" src="https://github.com/Weiyun1025/all-seeing/assets/79644233/e8876a63-6118-4319-abb7-409f025770f9">

Figure 1: Overview and comparison of our All-Seeing project with other popular large foundation models.

## Schedule
- [ ] Release the ASM model.
- [ ] Release the human verification results of AS-1B.
- [ ] Release the detailed region annotations of AS-1B.
- [ ] Release the semantic tags of AS-1B.
- [ ] Online demo, including dataset browser and ASM online demo.

## Online Demo
[**TODO**] The ASM model will be integrated into [InternGPT](https://github.com/OpenGVLab/InternGPT).

**Dataset Browser** will be available [here](todo).

## Dataset Overview
AS-1B with over 1 billion regions annotated with semantic tags, question-answering pairs, and detailed captions. It covers a wide range of 3.5 million common and rare concepts in the real world, and has 132.2 billion tokens that describe the concepts and their attributes.
<img width="800" alt="image" src="https://github.com/Weiyun1025/all-seeing/assets/8529570/88e89d79-8a46-4f42-af9e-1e8c23f0ba86">

Some examples

<img width="800" alt="image" src="https://github.com/Weiyun1025/all-seeing/assets/79644233/c0235f79-fbec-4cc1-b30d-d43f81ad1876">

## Model Architecture

The All-Seeing model (ASM) is a unified framework for panoptic visual recognition and understanding, including image/region-text retrieval, image/region recognition, captioning, and question-answering.
<img width="820" alt="image" src="https://github.com/Weiyun1025/all-seeing/assets/79644233/fb48668e-279c-484c-b22c-203102e26d4a">

## License

This project is released under the [Apache 2.0 license](LICENSE). 


<!-- ## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2023allseeing,
    title={The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World},
    author={Min Shi and Weiyun Wang and Qingyun Li and Wenhai Wang and Zhenhang Huang and Linjie Xing and Zhe Chen and Hao Li and Xizhou Zhu and Zhiguo Cao and Yushi Chen and Jifeng Dai and Yu Qiao},
    year={2023}
}
``` -->
