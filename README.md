
---
# NEEP: Neural Estimator for Entropy Production

[![Paper](http://img.shields.io/badge/paper-arxiv.2003.04166-B31B1B.svg)](https://arxiv.org/abs/2003.04166)
[![LICENSE](https://img.shields.io/github/license/kdkyum/neep.svg)](https://github.com/kdkyum/neep/blob/master/LICENSE)
![GitHub issues](https://img.shields.io/github/issues/kdkyum/neep.svg)
[![GitHub stars](https://img.shields.io/github/stars/kdkyum/neep.svg)](https://github.com/kdkyum/neep/stargazers)

## Introduction

This repo contains source code for the runs in [Learning entropy production via neural networks](https://arxiv.org/abs/2003.04166)

## Installation
```bash
git clone https://github.com/kdkyum/neep
cd neep
conda create -y --name neep python=3.6
conda activate neep
pip install -r requirements.txt
python -m ipykernel install --name neep
```

## Quickstart

```bash
jupyter notebook
```

See the following notebooks for the runs in the paper.
### Bead-spring model
* [`notebooks/bead-spring.ipynb`](notebooks/bead-spring.ipynb)

### Discrete flashing ratchet
* [`notebooks/ratchet.ipynb`](notebooks/ratchet.ipynb)

### RNEEP for Non-Markovian process
* [`notebooks/partial-ratchet-RNEEP.ipynb`](notebooks/partial-ratchet-RNEEP.ipynb)

## Usage

See option details by running following command
```
python train_bead_spring.py --help
```

### Command line running examples

* Bead-spring models (two- and five-bead model).

The training process is logged in `results/bead_spring` directory. Every training iteration of `record-freq`, 1,000 in this example, train loss (column name: "loss") and estimation of entropy production (EP) rate ("pred_rate") from training set are logged in `results/bead_spring/train_log.csv`. And test loss ("loss"), best test loss ("best_loss"), and estimation of EP rate ("pred_rate", "best_pred_rate") from test set are logged in `results/bead_spring/test_log.csv`.

```bash
python train_bead_spring.py \
  --Tc 1 \
  --Th 10 \
  --save results/bead_spring \
  --n-layer 3 \
  --n-hidden 256 \
  --n-bead 2 \
  --n-iter 100000 \
  --record-freq 1000 \
  --seed 5
```

* High-dimensional bead-spring models (N=8, 16, 32, 64, and 128).

```bash
python train_bead_spring_high.py \
  --save results/bead_spring_high \
  --n-layer 3 \
  --n-hidden 256 \
  --n-bead 8 \
  --n-iter 1000000 \
  --record-freq 10000 \
  --normalize \
  --seed 5
```

* Discrete flashing ratchet models.

```bash
python train_ratchet.py \
  --potential 2 \
  --save results/full_ratchet \
  --n-layer 1 \
  --n-hidden 128 \
  --n-iter 50000 \
  --record-freq 100 \
  --seed 5
```

* partial infromation ratchet model (RNEEP with sequence length `seq-len`).

```bash
python train_ratchet_partial.py \
  --potential 2 \
  --n-step 10000000 \
  --save results/partial_ratchet \
  --seq-len 32 \
  --n-layer 1 \
  --n-hidden 128 \
  --n-iter 100000 \
  --record-freq 1000 \
  --seed 5
```

## Author
Dong-Kyum Kim, Youngkyoung Bae, Sangyun Lee and Hawoong Jeong

## Bibtex
Cite following bibtex.
```bibtex
@article{kim2020learning,
  title={Learning entropy production via neural networks},
  author={Dong-Kyum Kim and Youngkyoung Bae and Sangyun Lee and Hawoong Jeong},
  journal={arXiv preprint arXiv:2003.04166},
  year={2020}
}
```

## License

This project following MIT License as written in LICENSE file.
