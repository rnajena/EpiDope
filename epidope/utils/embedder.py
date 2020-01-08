import numpy as np
import multiprocessing.pool
from allennlp.commands.elmo import ElmoEmbedder
import torch
from pathlib import Path


class Elmo_embedder():
    def __init__(self, model_dir="../elmo_settings", weights="weights.hdf5",
                 options="options.json", threads=1000):
        if threads == 1000:
            torch.set_num_threads(multiprocessing.cpu_count() // 2)
        else:
            torch.set_num_threads(threads)

        self.model_dir = Path(__file__).resolve().parent / Path(model_dir)
        self.weights = self.model_dir / weights
        self.options = self.model_dir / options
        self.seqvec = ElmoEmbedder(self.options, self.weights, cuda_device=-1)

    def elmo_embedding(self, x, start=None, stop=None):
        assert start is None and stop is None, "deprecated to use start stop, please trim seqs beforehand"

        if type(x[0]) == str:
            x = np.array([list(i.upper()) for i in x])
        embedding = self.seqvec.embed_sentences(x)
        X_parsed = []
        for i in embedding:
            X_parsed.append(i.mean(axis=0))
        return X_parsed


