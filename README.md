# Real-time Low-rank Recurrent Learning (UoM 2024)

This repo contains the Real-time Recurrent Learning (RTRL) implementation from the Master's thesis [Real-time Low-rank Recurrent Learning](https://drive.google.com/file/d/1cbxlMFmYMxHMe5p-0QS0sHMZGQXzZ-j3/view?usp=sharing), by Sehee Kim.

### Train
To run the code, follow the below.
```sh
python main.py --model [lstm or gru] --type [rtrl or lora_rtrl or snap or bptt] --level 1 --online true --recurrent-density 1 --inout-density 1
```
