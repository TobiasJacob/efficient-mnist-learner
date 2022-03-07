# Efficient-mnist-learner

**Problem Statement**: In most real-world applications, labelled data is scarce. Suppose you are given the Fashion-MNIST dataset (https://github.com/zalandoresearch/fashion-mnist), but without any labels in the training set. The labels are held in a database, which you may query to reveal the label of any particular image it contains. Your task is to build a classifier to >90% accuracy on the test set, using the smallest number of queries to this database.

**Approach**: Use a variational autoencoder for unsupervised training on the full unlabeled dataset. Then train the classifier on the encodings of the autoencoder.

### Demo

The encoded features:
![Demo video](docs/TSNEDemo.gif)

The images after reconstruction through the variational autoencoder:
![Reconstruction Demo](docs/ReconstructionDemo.png)

### Features

This project demonstrates

- `Convolutional variational autoencoder` in [Pytorch](https://pytorch.org/)
- `VS-Code dev container` with CUDA support
- [Pytorch-Lightning](https://www.pytorchlightning.ai/) for training. It would be possible to use vanilla Pytorch, however, if there exists a framework for it, it's better to use it.
- [Torchmetrics](https://torchmetrics.readthedocs.io/en/latest/) for calculating accuracy and f1-scores.
- [Hydra](https://hydra.cc/) for experiment management. All experiments are named according to the deviation from the default configuration.
- [Tensorboard](https://www.tensorflow.org/tensorboard) for advanced visualization of the encoded labels, and training statistics.
    - `Scalars` for training progress
    - `Images` for visualizing the reconstructed images from the autoencoder
    - `Projector` for visualizing the embeddings
- [Ax](https://ax.dev/) for hyperparameter optimization
- [flake8](https://flake8.pycqa.org/en/latest/) for linting, [black](https://black.readthedocs.io/en/stable/) for formatting, [isort](https://pycqa.github.io/isort/) for import sorting.

Architecture is inspired by [RDenseCNN](https://arxiv.org/pdf/2001.00526.pdf) and [SAM Optimizer](https://arxiv.org/abs/2010.01412).

Side Note: Usually, I use poetry or conda as a package manager. However, the NVIDIA-Docker container comes with an existing environment, that suffers from a `InvalidVersionSpec` error. Conda is also incredible slow in this case, therefore I am using `pip install` to add missing dependencies to the Docker Image. 

```console
(base) vscode@54a0b6598fe8:/workspaces/efficient-mnist-learner$ conda env export

InvalidVersionSpec: Invalid version '1.10.0+cu113<1.11.0': invalid character(s)
```

### Evaluation results

Number of training labels | Accuarcy on validation
--- | --- 
1000 | 84.74%
2000 | 86.25%
5000 | 89.37%
7500 | 90.42%
10000 | 91.03%
20000 | 91.65%
30000 | 93.04%
60000 | 93.56%

The auto-encoder results in a boost of 2-3% over a model which has not been pre-trained on the whole dataset.

## Setup GPU (tested on Ubuntu)

Open the project in the VS-Code dev container. Select the `base` conda environment that ships with the container. This is tested on CUDA Version 11.4, Ubuntu 20.04 LTS.

If you don't want to use VS-Code, you can run it in docker as well.

```console
cp requirements.txt .devcontainer/
docker build .devcontainer/ -t devcontainer
docker run --rm -it -v $(pwd):/workspace devcontainer bash
```

## Usage

You can run an experiment using the command line.

```console
python -m eml
```

Parameters can be overwritten the following way:

```console
python -m eml variational_sigma=0.001 unsupervised_epochs=10 classifier_epochs=10
```

More details can be found on the Hydra documentation. All configuration options are described in the [Config.py](eml/Config.py) file.
