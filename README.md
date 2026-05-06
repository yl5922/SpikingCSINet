# SNN-CSI-Feedback

Spiking neural networks for massive-MIMO CSI feedback.

This repo contains the training/evaluation code for our SNN-based CSI-feedback
model SpikingCSINet. Full training details and trained checkpoints will be
released alongside the paper.

## Models

Two models are provided (see `networks.py`):

- **`spikingcsinetpr`** — the proposed model. A progressive-residual SNN: at
  each spiking time step the encoder sees the current reconstruction residual
  and emits an additive correction, so reconstructions accumulate over `T`
  steps.
- **`spikingcsinet`** — ablation baseline. Same encoder/decoder blocks, but
  without the progressive-residual scheme: the encoder sees the original
  channel `x` at every step and per-step reconstructions are averaged.

Both share a convolutional encoder, a single LIF/IF spiking layer that
transmits binary spikes between encoder and decoder, and an FC decoder with a
linear skip path. Implementation builds on
[SpikingJelly](https://github.com/fangwei123456/spikingjelly).

## Repository layout

```
.
├── networks.py   # SpikingCSINet / SpikingCSINetPR
├── dataset.py    # COST2100 loaders, preprocessing, CSI augmentation
└── train.py      # training + evaluation entry point
```

## Requirements

Tested on Linux + NVIDIA GPU with the following versions:

```
python        3.10.16
torch         2.6.0+cu124   (CUDA 12.4)
numpy         2.2.4
scipy         1.15.2
spikingjelly  0.0.0.0.14
```

## Data

We use the standard COST2100 CSI dataset (the `.mat` files used by CsiNet and
follow-up work), which can be downloaded from
[Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA#list/path=%2F).
Place the files in a single directory, e.g. `./data/`:

```
data/
├── DATA_Htrainin.mat   DATA_Hvalin.mat   DATA_Htestin.mat     # indoor
└── DATA_Htrainout.mat  DATA_Hvalout.mat  DATA_Htestout.mat    # outdoor
```

Pass that directory via `--file_path`.

## Training

A single training run, e.g. CR=8, T=4, indoor:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model spikingcsinetpr --file_path ./data --envir indoor \
    --cr 8 --T 4 --enc_channels 4 --scale_factor 50 \
    --rx_hidden 8192 --skip_alpha 0.5 --skip_norm_align \
    --lr 0.002 --lr_schedule cosine --cosine_cycles 1 --epochs 1000 \
    --batch_size 128 --snn_dropout 0.0 \
    --step_mse_lambda 0.5 --step_mse_mode adaptive \
    --aug_phase_bins 16 --aug --seed 42 \
    --ckpt_dir ./checkpoints --save_state_dict --gpu_preload
```

Useful flags (see `train.py --help` for the full list):

- `--model {spikingcsinetpr, spikingcsinet}`
- `--envir {indoor, outdoor}`
- `--cr` — compression ratio (e.g. 4, 8, 16, 32, 64)
- `--T` — number of spiking time steps
- `--gpu_preload` — keep all data on GPU; eliminates CPU↔GPU transfer
- `--aug` — enable CSI phase-rotation augmentation

## Evaluation

```bash
python train.py --eval_only \
    --init_ckpt ./checkpoints/cr8_T4_spikingcsinetpr.pt \
    --model spikingcsinetpr --file_path ./data --envir indoor \
    --cr 8 --T 4 --enc_channels 4 --scale_factor 50 \
    --rx_hidden 8192 --skip_alpha 0.5 --skip_norm_align \
    --batch_size 128
```

This reports validation/test NMSE (in dB) and the average firing rate of the
spiking transmission layer, and writes a CSV row to `--eval_csv_dir` if
provided.

## Acknowledgement

This work would not have been possible without the open-source contributions
of the deep-learning-based CSI feedback community. We are
grateful to the authors of the following projects, whose codebases inspired
or informed parts of our implementation:

- **CRNet** — [Kylin9511/CRNet](https://github.com/Kylin9511/CRNet)
- **CLNet** — [SIJIEJI/CLNet](https://github.com/SIJIEJI/CLNet)
- **TransNet** — [Treedy2020/TransNet](https://github.com/Treedy2020/TransNet)

We also thank Chao-Kai Wen and Shi Jin's group for releasing the
pre-processed COST2100 dataset that the entire line of work builds on, and
the [SpikingJelly](https://github.com/fangwei123456/spikingjelly) team for
their SNN framework.
