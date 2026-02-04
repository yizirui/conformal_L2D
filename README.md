# Learning to Defer with an Uncertain Rejector via Conformal Prediction

Code for paper [Learning to Defer with an Uncertain Rejector via Conformal Prediction](https://openreview.net/forum?id=SZQJ8K2DUe) published in TMLR and published in [NeurIPS 2025 Workshop on Bayesian Decision-making and Uncertainty](https://openreview.net/forum?id=TWb9y4PNSW).



## What is it?


Learning to defer (L2D) allows prediction tasks to be allocated to a human or machine decision maker, thus getting the best of both’s abilities. Yet this allocation decision depends on a'rejector’ function, which could be poorly fit or otherwise misspecified. In this work, we perform uncertainty quantification for the rejector subcomponent of the L2D framework. We use conformal prediction to allow the reject to output sets instead of just the binary outcome of ‘defer’ or not. On tasks ranging from object to hate speech detection, we demonstrate that the uncertainty in the rejector translates to safer decisions via two forms of selective prediction.

As a motivating example, in the diagnosis of skin lesions using the HAM10000 dataset, an abstention indicates that the current data does not provide sufficient confidence for a reliable prediction. In such scenarios, decision-makers can seek further information, such as laboratory tests, to ensure more accurate and safer outcomes.

This strategy aligns with practices in medical diagnostics, where clinicians often require additional data before making definitive decisions, thereby enhancing the reliability of diagnoses and treatments.


## Structure and Project Organization

![](datasets.png)

This repository is organized as follows:

- datasetsdefer: implementation of the different datasets used, each as a seperate class

- methods: proposed RealizableSurrogate and MILP 

- conformal_expers: replicates main experiments structured by each dataset
    - `CIFAR-10`: This folder contains the experiments for CIFAR-10 dataset
    - `HAM10k`: This folder containes the experiments for HAM10000 dataset
    - `Hatespeech`: This folder contains the experiments for Hatespeech dataset. 

- networks: PyTorch model class definitions for linear, ResNet34, CNN and FFNN models

- helpers: various methods used, including metrics implementation

One can easily implement new surrogate methods by extending [BaseSurrogateMethod](baselines/basemethod.py). 



## Installation

Create a new conda environmnt (Python 3.8.5 with Pytorch 1.7.1) from our environment.yml file:
```bash
conda env create -f environment.yml
```

Finally activate the environment:
```bash
conda activate deferral
```

And clone the repo:
```bash
git clone https://github.com/yizirui/conformal_L2D
```


## Setup
The experiment scripts are portable and accept CLI flags or environment variables for paths.
Required setup:
- Dataset files under `data/` (or set `L2D_DATA_DIR` to point elsewhere).
- `config/columns.json` for CSV output schema (or set `L2D_COLUMNS_PATH`).

Optional environment variables (override defaults):
```bash
export L2D_PROJECT_ROOT=/path/to/repo   # defaults to repo root
export L2D_DATA_DIR=/path/to/data       # defaults to $L2D_PROJECT_ROOT/data
export L2D_RESULTS_DIR=/path/to/results # defaults to $L2D_PROJECT_ROOT/output/expers
export L2D_COLUMNS_PATH=/path/to/columns.json
```

## Experiments
Set up experiment variables

```bash
EXPER_NAME=""
ALPHA=0.1
```
If you run from outside the repo, set `L2D_PROJECT_ROOT` or pass `--data-dir`, `--results-dir`, `--columns-path` directly.
### CIFAR10

#### Major Experiment
`synth_conformal_corrupted.py`

```bash
EXPER_NAME="cifar10_major"
ALPHA=0.1
SUROGATE="OvA"
MODEL="SimpleCNN"

python conformal_expers/cifar10/synth_conformal_corrupted.py \\
  -d False \\
  -ep $EXPER_NAME \\
  -dm ConformalAndThreshold \\
  -m $MODEL \\
  -s $SUROGATE \\
  -a $ALPHA \\
  -r 42
```

Our experiments are on `corruption_types =  ['brightness']`. Other corruption types can be selected with `--corruptions` (comma-separated), for example:
```python
corruption_types = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
    'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
    'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
    'zoom_blur'
]
```

#### Binary Regression
`conformal_expers/cifar10/binary_reg_synth_conformal_corrupted.py`

```bash
EXPER_NAME="cifar10_major"
ALPHA=0.1
SUROGATE="OvA"
MODEL="SimpleCNN"
python conformal_expers/cifar10/binary_reg_synth_conformal_corrupted.py -d True -ep $EXPER_NAME  -dm ConformalAndThreshold -m $MODEL -s $SUROGATE -a $ALPHA -r 42  
```

### HAM10000

#### Major Experiment
`conformal_expers/HAM10k/synth_conformal_corrupted_ham10000.py`

```bash
EXPER_NAME="ham10000_major"
ALPHA=0.1
SUROGATE="OvA"
MODEL="ResNet34"

python conformal_expers/HAM10k/synth_conformal_corrupted_ham10000.py \\
  -d False \\
  -ep $EXPER_NAME \\
  -dm ConformalAndThreshold \\
  -m $MODEL \\
  -s $SUROGATE \\
  -a $ALPHA \\
  -r 42
```


### Hate Speech

#### Major Experiment
`conformal_expers/hate_speech/synth_conformal_corrupted_hate.py`

```bash
EXPER_NAME="hate_speech_major"
ALPHA=0.1
SUROGATE="OvA"
MODEL="LinearNet"

python conformal_expers/hate_speech/synth_conformal_corrupted_hate.py \\
  -d False \\
  -ep $EXPER_NAME \\
  -dm ConformalAndThreshold \\
  -m $MODEL \\
  -s $SUROGATE \\
  -a $ALPHA \\
  -r 42
```


# Surrogate Losses
We incorrperate a host of methods from the literature as well as our new methods:

## OvASurrogate 
One-over-all by (Verma and Nalisnick 2022): `
SUROGATE="OvA"`

## SelectivePrediction: 
Classifier is trained on cross-entropy (CE), rejector tresholds confidence of classifier. Use variable `SUROGATE="SP"` 

## ALCE: 
Asymmetric softmax by (Cao et al., 2023):
`SUROGATE=ALCE`



# Citation

```
@article{
fang2025learning,
title={Learning to Defer with an Uncertain Rejector via Conformal Prediction},
author={Yizirui Fang and Eric Nalisnick},
journal={Transactions on Machine Learning Research},
year={2025},
url={https://openreview.net/forum?id=SZQJ8K2DUe},
note={}
}

@inproceedings{fang2024learning,
  title={Learning to Defer with an Uncertain Rejector via Conformal Prediction},
  author={Fang, Yizirui and Nalisnick, Eric},
  booktitle={NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty}
}
```
