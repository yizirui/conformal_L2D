# Data preparation

Run these commands from the repository root. The runner reads from `data/` by default. Use `--data-dir` to select a different directory.

## CIFAR-10 and SVHN

Torchvision downloads clean CIFAR-10 automatically. The split is 42,000 training, 6,000 calibration and 12,000 test images. The test split includes all 10,000 originals used by CIFAR-10-C.

For Figure 2, extract the brightness corruption (levels 1–5) and labels from CIFAR-10-C. [Dataset archive](https://zenodo.org/records/2535967)

```bash
mkdir -p data
curl -fL 'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1' -o data/CIFAR-10-C.tar
tar -xf data/CIFAR-10-C.tar -C data CIFAR-10-C/brightness.npy CIFAR-10-C/labels.npy
```

At level 6, torchvision downloads the SVHN train and test files. The runner uses 7,325 training digits for target calibration and all 26,032 test digits for evaluation. On CIFAR-10, CIFAR-10-C and SVHN, the simulated expert returns the true label for classes 0–5 and guesses uniformly among all ten labels otherwise. [SVHN files](https://ufldl.stanford.edu/housenumbers/) · [Upstream expert](https://github.com/clinicalml/human_ai_deferral/blob/94c05bcdb6822009835fe051dc11a0a878f3e6f4/datasetsdefer/cifar_synth.py)

## HAM10000

Place `HAM10000_metadata.csv` in `data/HAM10000/` and the JPEGs from both archives in `data/HAM10000/HAM10000_images/`. Prepare the 60%/20%/20% training, calibration and test splits, then download the fixed metadata expert used to simulate the annotator. [Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) · [Multi_L2D expert](https://github.com/rajevv/Multi_L2D/blob/3f2ad6bc93afdfd250faac21225f6b55074b6cf8/ham10000/MLP_Mixer_model/m_expert)

```bash
python scripts/preprocess_ham10000.py data/HAM10000
curl -fL --create-dirs -o datasetsdefer/ham10000/MLP_Mixer_model/m_expert \
  https://raw.githubusercontent.com/rajevv/Multi_L2D/3f2ad6bc93afdfd250faac21225f6b55074b6cf8/ham10000/MLP_Mixer_model/m_expert
```

The runner trains the L2D model from an ImageNet-initialized ResNet34. Severity changes the pixel mask and blur applied to images. [Shift settings](../datasetsdefer/shift_transforms.py)

## Hate Speech

The loader downloads the labeled tweets and multilingual MiniLM encoder on first use. It splits tweets 70%/10%/20% and samples one expert response per tweet using the proportions of its three annotation counts. Responses are cached by annotator and seed, so each severity uses the same splits and expert answers. [Labeled tweets](https://github.com/t-davidson/hate-speech-and-offensive-language) · [Encoder](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) · [Embedding perturbations](../datasetsdefer/hatespeech.py)
