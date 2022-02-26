# Efficient-mnist-learner

**Problem Statement**: In most real-world applications, labelled data is scarce. Suppose you are given the Fashion-MNIST dataset (https://github.com/zalandoresearch/fashion-mnist), but without any labels in the training set. The labels are held in a database, which you may query to reveal the label of any particular image it contains. Your task is to build a classifier to >90% accuracy on the test set, using the smallest number of queries to this database.

**Approach**: Use a variational autoencoder for unsupervised training on the full unlabeled dataset. Then train the classifier on the encodings of the autoencoder.

### Features

![Demo video](docs/TSNEDemo.gif)

This project demonstrates

- `Convolutional variational autoencoder` in Pytorch
- `VS-Code dev container` with CUDA support
- `Pytorch-Lightning` for training. It would be possible to use vanilla Pytorch, however, if there exists a framework for it, it's better to use it.
- `Torchmetrics` for calculating accuracy and f1-scores.
- `Hydra` for experiment management. All experiments are named according to the deviation from the default configuration.
- `Tensorboard` for advanced visualization of the encoded labels, and training statistics.
    - `Scalars` for training progress
    - `Images` for visualizing the reconstructed images from the autoencoder
    - `Projector` for visualizing the embeddings
- `Ax` for hyperparameter optimization

Side Note: Usually, I use poetry or conda as a package manager. However, the NVIDIA-Docker container comes with an existing environment, that suffers from an `InvalidVersionSpec` error. Conda is also incredible slow in this case, therefore I am using `pip install` to add missing dependencies to the Docker-Image. 

```console
(base) vscode@54a0b6598fe8:/workspaces/efficient-mnist-learner$ conda env export

InvalidVersionSpec: Invalid version '1.10.0+cu113<1.11.0': invalid character(s)
```

## Setup GPU (tested on Ubuntu)

Open the project in the VS-Code dev container. Select the `base` conda environment that ships with the container. This is tested on CUDA Version 11.4, Ubuntu 20.04 LTS.

## Usage

You can use any 

```console
python -m eml 

```

## For t-sne: Use this

Perplexity: 76
LR: 0.1
Supervise: 0
