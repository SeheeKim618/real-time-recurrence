# Real-time Low-rank Recurrent Learning (UoM 2024)

This repo contains the Real-time Recurrent Learning (RTRL) implementation from the Master's thesis [Real-time Low-rank Recurrent Learning](https://drive.google.com/file/d/1cbxlMFmYMxHMe5p-0QS0sHMZGQXzZ-j3/view?usp=sharing), by Sehee Kim.

## Introduction
Recurrent Neural Networks (RNNs) are designed to learn long-term dependencies in sequential data. The most widely used training method for RNNs, Backpropagation Through Time (BPTT), requires storing a complete history of network states, which prevents online weight updates after each timestep. In contrast, Real-Time Recurrent Learning (RTRL) enables online updates without storing the entire history using forward-mode differentiation. However, the computational cost of RTRL scales quadratically with the state size, making it impractical for even moderately sized RNN models. Recent approaches have attempted to reduce these costs by providing sparse approximations to RTRL. However, they have not achieved the same level of performance as RTRL and still require similar training times. This thesis introduces a novel and efficient approximation for RTRL based on Low-Rank Adaptation (LoRA). We show that applying low-rank adaptation to RTRL achieves a 57.9\% faster training time than standard RTRL and reduces the number of Jacobian parameters by up to 1.9 times without compromising performance. These reductions become more significant as the number of hidden layers increases. Additionally, we find that varying the rank for different components of weights in neural networks is more effective than using a fixed rank, as in the original LoRA method. We empirically compare our approach to recent RTRL algorithms, such as the Sparse n-step Approximation (SnAp), by reproducing their results. Our method reduces training time by up to 50\% while achieving better performance than SnAp-1.

## Environment
The code is developed using Jax and python 3.10. GPUs are not necessarily needed. We developed and tested using a single NVIDIA V100 GPU card.

## Quick start
### Installation
Install dependencies:
   ```
   pip install -r requirements.txt
   ```
### Train
To run the code, follow the below.
```sh
python main.py --model [lstm or gru] --type [rtrl or lora_rtrl or snap or bptt] --level 1 --online true --recurrent-density 1 --inout-density 1
```

## Citation
````
@inproceedings{
menick2021practical,
title={Practical Real Time Recurrent Learning with a Sparse Approximation},
author={Jacob Menick and Erich Elsen and Utku Evci and Simon Osindero and Karen Simonyan and Alex Graves},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=q3KSThy2GwB}
}
````
