# UNO

This repository provides a reference implementation of the paper: *Enhancing Fine-Grained Urban Flow Inference via Incremental Neural Operator*

## Requirements
We implement ENHANCER and other FUFI methods with the following dependencies:
* python 3.11.5
* pytorch 2.1.2
* numpy
* einops
* scikit-learn

## Datasets
TaxiBJ datasets can be obtained from the baseline [UrbanFM's repository](https://github.com/yoshall/UrbanFM/tree/master/data).

## Usage
Before running the code, ensure the package structure of ENHANCER is as follows:
```
.
├── datasets
│   └── TaxiBJ
│       ├── P1
│       ├── P2
│       ├── P3
│       └── P4
├── experiments
├── model
├── uno_train
└── utils_pack
```



## Citing
If you find UNO is useful in your research, please cite the following paper:
```bibtex
@inproceedings{
  title={Enhancing Fine-Grained Urban Flow Inference via Incremental Neural Operator},
  author={Qiang Gao, Xiaolong Song, Li Huang, Goce Trajcevski2, Fan Zhou and Xueqin Chen},
  booktitle={IJCAI},
  year={2024}
}
```