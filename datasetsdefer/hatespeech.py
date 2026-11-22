import logging
import os
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer

from .basedataset import BaseDataset


# Section 5.3 uses the same per-example dimension fraction for masking and edits.
_DIMENSION_FRACTION = {0: 0.0, 1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.40}
_JITTER_STD = {0: 0.0, 1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.10, 6: 0.15}


def _save_cached_tensor(tensor, path):
    """Make the complete tensor visible at once to concurrent experiment runs."""
    with TemporaryDirectory(dir=path.parent) as tmp_dir:
        temporary = Path(tmp_dir) / path.name
        torch.save(tensor, temporary)
        temporary.replace(path)


class ModelPredictAAE:
    def __init__(self, modelfile, vocabfile):
        """
        Edited from https://github.com/slanglab/twitteraae
        """
        self.vocabfile = vocabfile
        self.modelfile = modelfile
        self.load_model()

    def load_model(self):

        self.N_wk = np.loadtxt(self.modelfile)
        self.N_k = self.N_wk.sum(0)
        self.K = len(self.N_k)
        self.wordprobs = (self.N_wk + 1) / self.N_k
        self.vocab = [
            L.split("\t")[-1].strip() for L in open(self.vocabfile, encoding="utf8")
        ]
        self.w2num = {w: i for i, w in enumerate(self.vocab)}
        assert len(self.vocab) == self.N_wk.shape[0]

    def infer_cvb0(self, invocab_tokens, alpha, numpasses):
        doclen = len(invocab_tokens)

        # initialize with likelihoods
        Qs = np.zeros((doclen, self.K))
        for i in range(0, doclen):
            w = invocab_tokens[i]
            Qs[i, :] = self.wordprobs[self.w2num[w], :]
            Qs[i, :] /= Qs[i, :].sum()
        lik = Qs.copy()  # pertoken normalized but proportionally the same for inference

        Q_k = Qs.sum(0)
        for _ in range(1, numpasses):
            for i in range(0, doclen):
                Q_k -= Qs[i, :]
                Qs[i, :] = lik[i, :] * (Q_k + alpha)
                Qs[i, :] /= Qs[i, :].sum()
                Q_k += Qs[i, :]

        Q_k /= Q_k.sum()
        return Q_k

    def predict_lang(self, tokens, alpha=1, numpasses=5, thresh1=1, thresh2=0.2):
        invocab_tokens = [w.lower() for w in tokens if w.lower() in self.w2num]
        # check that at least xx tokens are in vocabulary
        if len(invocab_tokens) < thresh1:
            return None
        # check that at least yy% of tokens are in vocabulary
        elif len(invocab_tokens) / len(tokens) < thresh2:
            return None
        else:
            posterior = self.infer_cvb0(
                invocab_tokens, alpha=alpha, numpasses=numpasses
            )
            # posterior is probability for African-American, Hispanic, Asian, and White (in that order)
            aae = (np.argmax(posterior) == 0) * 1
            return aae


class HateSpeech(BaseDataset):
    """Embedded Davidson et al. tweets with sampled or synthetic expert labels."""

    def __init__(
        self,
        data_dir,
        embed_texts,
        include_demographics,
        expert_type,
        device,
        severity=0,
        synth_exp_param=(0.7, 0.7),
        test_split=0.2,
        val_split=0.1,
        batch_size=1000,
        transforms=None,
        split_seed=None,
        random_seed=None,
    ):
        if not embed_texts:
            raise ValueError("Hate Speech experiments require sentence embeddings.")
        self.expert_type = expert_type
        self.synth_exp_param = synth_exp_param
        self.data_dir = data_dir
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 3
        self.split_seed = split_seed
        self.random_seed = random_seed
        self.dataset_name = f"hatespeech_{severity}"
        self.generate_data(severity)

    def _synthetic_annotations(self, tweets, rng):
        for filename in ("model_count_table.txt", "model_vocab.txt"):
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                logging.info("Downloading TwitterAAE %s", filename)
                response = requests.get(f"https://github.com/slanglab/twitteraae/raw/master/model/{filename}")
                response.raise_for_status()
                with open(path, "wb") as output:
                    output.write(response.content)
        model = ModelPredictAAE(
            os.path.join(self.data_dir, "model_count_table.txt"),
            os.path.join(self.data_dir, "model_vocab.txt"),
        )
        demographics = tweets["tweet"].apply(model.predict_lang)
        predictions = []
        for group, label in zip(demographics, tweets["class"]):
            probability = self.synth_exp_param[0 if group == 0 else 1]
            correct = rng.choice([0, 1], p=[1 - probability, probability])
            predictions.append(label if correct else rng.choice([0, 1, 2]))
        return predictions

    def _expert_labels(self, tweets, cache_dir):
        """Sample one response per tweet and reuse it across all shift severities."""
        path = None
        rng = np.random
        if self.random_seed is not None:
            directory = cache_dir / "annotations" / "random_annotator"
            if self.expert_type == "synthetic":
                rates = "_".join(str(float(p)) for p in self.synth_exp_param)
                directory = cache_dir / "annotations" / "synthetic" / rates
            path = directory / f"seed_{self.random_seed}.pt"
            if path.is_file():
                return torch.load(path)
            directory.mkdir(parents=True, exist_ok=True)
            rng = np.random.RandomState(self.random_seed)

        if self.expert_type == "synthetic":
            predictions = self._synthetic_annotations(tweets, rng)
        else:
            counts = tweets[["hate_speech", "offensive_language", "neither"]].to_numpy()
            predictions = [rng.choice([0, 1, 2], p=row / sum(row)) for row in counts]
        labels = torch.as_tensor(predictions)
        if path is not None:
            _save_cached_tensor(labels, path)
        return labels

    def generate_data(self, severity=0):
        path = os.path.join(self.data_dir, "hatespeech_labeled_data.csv")
        if not os.path.exists(path):
            logging.info("Downloading Hate Speech dataset")
            response = requests.get(
                "https://github.com/t-davidson/hate-speech-and-offensive-language/raw/master/data/labeled_data.csv"
            )
            response.raise_for_status()
            with open(path, "wb") as output:
                output.write(response.content)
        hatespeech_data = pd.read_csv(path)
        cache_dir = Path(self.data_dir) / "hatespeech"
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_h = self._expert_labels(hatespeech_data, cache_dir)
        train_y = torch.from_numpy(hatespeech_data["class"].to_numpy())
        split_seed = random.randrange(10000) if self.split_seed is None else self.split_seed

        # Initialize the encoder on both cold and warm caches to preserve the training RNG sequence.
        encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = encoder.encode(hatespeech_data["tweet"].to_numpy())
        train_x = torch.from_numpy(np.array(embeddings)).float()

        train_x_path = cache_dir / "train_x.pt"
        train_y_path = cache_dir / "train_y.pt"
        if train_x_path.is_file() and train_y_path.is_file():
            logging.info("Loading cached embeddings")
            train_x = torch.load(train_x_path)
            train_y = torch.load(train_y_path)
        else:
            _save_cached_tensor(train_x, train_x_path)
            _save_cached_tensor(train_y, train_y_path)

        self.d = train_x.shape[1]
        train_x = self.apply_adversarial_noise(train_x, severity)

        test_size = int(self.test_split * len(train_x))
        val_size = int(self.val_split * len(train_x))
        train_size = len(train_x) - test_size - val_size
        rows = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(split_seed))
        train, val, test = rows.split((train_size, val_size, test_size))

        def split_dataset(indices):
            return torch.utils.data.TensorDataset(train_x[indices], train_y[indices], train_h[indices])

        self.data_train_loader = torch.utils.data.DataLoader(
            split_dataset(train), batch_size=self.batch_size, shuffle=True
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            split_dataset(val), batch_size=self.batch_size, shuffle=False
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            split_dataset(test), batch_size=self.batch_size, shuffle=False
        )

    def remove_dimensions(self, x, level):
        """Mask a randomly selected fraction of each embedding's dimensions in place."""
        num_features = x.shape[1]
        num_to_zero = int(num_features * _DIMENSION_FRACTION[level])
        for row in x:
            indices = torch.randperm(num_features, device=x.device)[:num_to_zero]
            row[indices] = 0
        return x

    def apply_adversarial_noise(self, x, level):
        """Apply the embedding jitter, token-inspired edits and dimension mask of §5.3."""
        if level >= 1:
            x.add_(torch.randn_like(x), alpha=_JITTER_STD[level])
            x = self.remove_dimensions(x, level)
            x = self.character_swap(x, level)
        # Mask last so the selected coordinates stay zero after jitter and swaps.
        return self.remove_dimensions(x, level)

    def character_swap(self, x, level):
        swap_amount = int(x.shape[1] * _DIMENSION_FRACTION[level])
        for row in x:
            indices = torch.randperm(x.shape[1], device=x.device)[:swap_amount]
            row[indices] = row[indices].roll(shifts=1)
        return x
