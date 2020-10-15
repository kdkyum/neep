
---
# NEEP: Neural Estimator for Entropy Production

[![PRL](http://img.shields.io/badge/PRL-PhysRevLett.125.140604-517E66.svg)](https://link.aps.org/doi/10.1103/PhysRevLett.125.140604)
[![arxiv](http://img.shields.io/badge/arXiv-2003.04166-B31B1B.svg)](https://arxiv.org/abs/2003.04166)
[![LICENSE](https://img.shields.io/github/license/kdkyum/neep.svg)](https://github.com/kdkyum/neep/blob/master/LICENSE)

Authors: Dong-Kyum Kim<sup>1*</sup>, Youngkyoung Bae<sup>1*</sup>, Sangyun Lee<sup>1</sup> and Hawoong Jeong<sup>1,2</sup><br>
<sub>\* Equal contribution</sub>

<sup>1</sup> <sub>Department of Physics, KAIST</sub>
<sup>2</sup> <sub>Center for Complex Systems, KAIST</sub>

## Summary

We developed the neural estimator for entropy production (NEEP), a novel type of estimator that evaluates entropy production from the trajectories of system variables by deep learning. In the macroscopic world, entropy never decreases as time moves forward; for instance, spilled water will not spontaneously jump back into its cup. However, entropy can in fact transiently decrease in small systems due to environmental fluctuations. Physicists have found that such an entropy difference, called *entropy production* (EP), can be measured by estimating the probability that a time-reversed trajectory, which is like playing a recorded video from an experiment backward, will be observed. Various EP estimators based on this method have been developed, but the required trajectory data needs to first be divided into discrete microstates. In contrast to these conventional methods, NEEP does not require such detailed information to estimate the probability of time-reversed trajectories, and further, NEEP can efficiently learn EP from data even in high-dimensional space. We additionally verified that our approach is even applicable to time-series data with unobservable variables. We expect NEEP to be beneficial in applications to intricately organized systems where numerous variables are entangled and hidden, such as biological systems, active matter, and others, for a deeper look into their complex dynamics.

<center><img src="neep.png" width="600px"/></center>

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

See option details by running the following command
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

* partial information ratchet model (RNEEP with sequence length `seq-len`).

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

## Bibtex
Cite the following Bibtex.
```bibtex
@article{kim2020learning,
  title = {Learning Entropy Production via Neural Networks},
  author = {Kim, Dong-Kyum and Bae, Youngkyoung and Lee, Sangyun and Jeong, Hawoong},
  journal = {Phys. Rev. Lett.},
  volume = {125},
  issue = {14},
  pages = {140604},
  numpages = {6},
  year = {2020},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.125.140604},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.125.140604}
}
```

## License

This project following the MIT license.
