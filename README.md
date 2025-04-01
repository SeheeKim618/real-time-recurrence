# Real-time Low-rank Recurrent Learning (UoM 2024)

This repo contains the Real-time Recurrent Learning (RTRL) implementation from the Master's thesis [Real-time Low-rank Recurrent Learning](https://drive.google.com/file/d/1cbxlMFmYMxHMe5p-0QS0sHMZGQXzZ-j3/view?usp=sharing), by Sehee Kim.

## Introduction
Recurrent Neural Networks (RNNs) are designed to learn long-term dependencies in sequential data. The most widely used training method for RNNs, Backpropagation Through Time (BPTT), requires storing a complete history of network states, which prevents online weight updates after each timestep. In contrast, Real-Time Recurrent Learning (RTRL) enables online updates without storing the entire history using forward-mode differentiation. However, the computational cost of RTRL scales quadratically with the state size, making it impractical for even moderately sized RNN models. Recent approaches have attempted to reduce these costs by providing sparse approximations to RTRL. However, they have not achieved the same level of performance as RTRL and still require similar training times. This thesis introduces a novel and efficient approximation for RTRL based on Low-Rank Adaptation (LoRA). We show that applying low-rank adaptation to RTRL achieves a 57.9\% faster training time than standard RTRL and reduces the number of Jacobian parameters by up to 1.9 times without compromising performance. These reductions become more significant as the number of hidden layers increases. Additionally, we find that varying the rank for different components of weights in neural networks is more effective than using a fixed rank, as in the original LoRA method. We empirically compare our approach to recent RTRL algorithms, such as the Sparse n-step Approximation (SnAp), by reproducing their results. Our method reduces training time by up to 50\% while achieving better performance than SnAp-1.

For RTRL, the Jacobian matrix $J_t$ is defined as:

$$J_t = I_t + D_t J_{t-1}$$  

We decompose the influence Jacobian $J_t$ - including the immediate Jacobian $I_t$, and the influence Jacobian from the previous timestep $J_{t-1}$ - into two low-rank matrices, $J_A$ and $J_B$. Thus, the low-rank Jacobian can be expressed as:

$$J_{Bt}$$

J_{A_{<t}} = J_{B_t} J_{A_t} + D_t J_{B_{<t-1}}^{-1} J_{A_{<t-1}}

For GRU, the update rules are also modified to include low-rank terms. The standard
GRU equations are: 

$$z_t = \sigma\Bigl(W_z x_t + U_z h_{t-1} + b_z\Bigr),$$  
$$r_t = \sigma\Bigl(W_r x_t + U_r h_{t-1} + b_r\Bigr),$$  
$$\tilde{h_t} = \tanh\Bigl(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h\Bigr),$$  
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$  

In the modified GRU:

$$z_t = \sigma\Bigl(\bigl(W_z + W_z^A W_z^B\bigr)x_t + \bigl(U_z + U_z^A U_z^B\bigr)h_{t-1} + b_z\Bigr),$$  
$$r_t = \sigma\bigl(\bigl(W_r + W_r^A W_r^B\bigr)x_t + \bigl(U_r + U_r^A U_r^B\bigr)h_{t-1} + b_r\bigr),$$  
$$\tilde{h_t} = \tanh\Bigl(\bigl(W_h + W_h^A W_h^B\bigr)x_t + \bigl(U_h + U_h^A U_h^B\bigr)\bigl(r_t \odot h_{t-1}\bigr) + b_h\Bigr),$$  
$$h_t = \bigl(1 - z_t\bigr)\odot h_{t-1} + z_t \odot \tilde{h}_t$$  



## Benchmarking
### Comparison of the validation loss on copy task 
![Illustrating the performance](/figures/result1.png)


### Comparison of the number of Jacobian parameters on copy task 
|          | RTRL | SnAp-1 | SnAp-2 (d = 0.6) | LoRA-RTRL (r = 2,8) |
|--------------------|----------|------------|---------|--------|
| LSTM       | 143,360  | 4,480      |  83,927  | 74,496   | 
| GRU               | 107,520  | 3,360      |  63,881  | 55,872  |

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
@article{hu2022lora,
  title={Lora: Low-rank adaptation of large language models.},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu and others},
  journal={ICLR},
  volume={1},
  number={2},
  pages={3},
  year={2022}
}

@inproceedings{
menick2021practical,
title={Practical Real Time Recurrent Learning with a Sparse Approximation},
author={Jacob Menick and Erich Elsen and Utku Evci and Simon Osindero and Karen Simonyan and Alex Graves},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=q3KSThy2GwB}
}
````
