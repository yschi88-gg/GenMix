# GenMix

Official PyTorch implementation of **GenMix: Self-Supervised Contrastive Learning via Adaptive Feature Mixing and Generative Latent Augmentation**.

## Files

```text
GenMix/
├── README.md
├── requirements.txt
├── main-cifar10.py
├── main-cifar100.py
└── imagenet100.py
```

## Installation

```bash
conda create -n genmix python=3.10 pip -y
conda activate genmix
```

Install PyTorch and torchvision for your CUDA version using the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then run:

```bash
pip install -r requirements.txt
```

## Datasets

- [CIFAR-10 / CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet-100](https://www.kaggle.com/datasets/ambityga/imagenet100)

CIFAR-100 downloads automatically. Prepare CIFAR-10 before training:

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='./data', train=True, download=True)"
```

For ImageNet-100, arrange the downloaded images as follows, with matching class-folder names in both splits:

```text
data/imagenet100/
├── train/
│   ├── class_1/
│   └── ...
└── val/
    ├── class_1/
    └── ...
```

## Run

Run from the repository directory. Each script performs pretraining followed by linear evaluation. Include `--use_clsp` to enable GLA.

### CIFAR-10

```bash
python -u main-cifar10.py --device cuda --use_clsp --data_dir ./data --exp_dir ./logs/cifar10 --epochs_uns 800 --epochs_sup 800 --batch_size 512 --lr_uns 0.06 --lr_sup 30 --lambda_syn 0.4 --seed 0
```

### CIFAR-100

```bash
python -u main-cifar100.py --device cuda --use_clsp --data_dir ./data --exp_dir ./logs/cifar100 --epochs_uns 800 --epochs_sup 800 --batch_size 512 --lr_uns 0.06 --lr_sup 30 --lambda_syn 0.4 --seed 0
```

### ImageNet-100

```bash
python -u imagenet100.py --device cuda --use_clsp --data_dir ./data/imagenet100 --exp_dir ./logs/imagenet100 --epochs_uns 400 --epochs_sup 100 --batch_size 128 --lr_uns 0.5 --lr_sup 30 --lambda_syn 0.4 --seed 0
```
