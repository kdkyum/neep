# NEEP: Neural Estimator for Entropy Production

[![LICENSE](https://img.shields.io/github/license/kdkyum/neep.svg)](https://github.com/kdkyum/neep/blob/master/LICENSE)
![GitHub issues](https://img.shields.io/github/issues/kdkyum/neep.svg)
[![GitHub stars](https://img.shields.io/github/stars/kdkyum/neep.svg)](https://github.com/kdkyum/neep/stargazers)

## Introduction

This repo is implementation of NEEP.

## Installation
```bash
git clone https://github.com/kdkyum/neep
cd neep
conda env create -f environment.yml
conda activate neep
python -m ipykernel install --name neep
```

## Quickstart

```bash
jupyter notebook
```

### Bead-spring model
See our notebook file at [`notebooks/bead-spring.ipynb`](notebooks/bead-spring.ipynb)

### Discrete flashing ratchet
* [ ] To be soon

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
