from typing import *
import json

import numpy as np

import asm2vec.asm
import asm2vec.repo

import asm2vec.internal.training
import asm2vec.internal.representative
import asm2vec.internal.serialization
import asm2vec.internal.util


class Asm2VecMemento:
    def __init__(self):
        self.params: Optional[asm2vec.internal.training.Asm2VecParams] = None
        self.vocab: Optional[Dict[str, asm2vec.repo.Token]] = None

    def serialize(self, fp) -> None:
        d = {
            'params': self.params.to_dict(),
            'vocab': asm2vec.internal.serialization.serialize_vocabulary(self.vocab)
        }
        json.dump(d, fp)

    def populate(self, fp) -> None:
        d = json.load(fp)
        self.params = asm2vec.internal.training.Asm2VecParams(**d['params'])
        self.vocab = asm2vec.internal.serialization.deserialize_vocabulary(self.vocab)


class Asm2Vec:
    def __init__(self, **kwargs):
        self._params = asm2vec.internal.training.Asm2VecParams(**kwargs)
        self._vocab = None

    def memento(self) -> Asm2VecMemento:
        memento = Asm2VecMemento()
        memento.params = self._params
        memento.vocab = self._vocab
        return memento

    def set_memento(self, memento: Asm2VecMemento) -> None:
        self._params = memento.params
        self._vocab = memento.vocab

    def make_function_repo(self, funcs: List[asm2vec.asm.Function]) -> asm2vec.repo.FunctionRepository:
        return asm2vec.internal.representative.make_function_repo(
            funcs, self._params.d, self._params.num_of_rnd_walks, self._params.jobs)

    def train(self, repo: asm2vec.repo.FunctionRepository) -> None:
        asm2vec.internal.training.train(repo, self._params)
        self._vocab = repo.vocab()

    def to_vec(self, f: asm2vec.asm.Function) -> np.ndarray:
        estimate_repo = asm2vec.internal.representative.make_estimate_repo(
            self._vocab, f, self._params.d, self._params.num_of_rnd_walks)
        vf = estimate_repo.funcs()[0]

        asm2vec.internal.training.estimate(vf, estimate_repo, self._params)

        return vf.v
