from typing import *
import json

import numpy as np

import asm2vec.internal.training
import asm2vec.asm
import asm2vec.internal.representative
import asm2vec.internal.serialization
import asm2vec.internal.utilities


class Asm2VecMemento:
    def __init__(self):
        self.params: Optional[asm2vec.internal.training.Asm2VecParams] = None
        self.vocab: Optional[Dict[str, asm2vec.internal.representative.Token]] = None
        self.num_of_tokens: Optional[int] = None

    def serialize(self, fp) -> None:
        d = {
            'params': self.params.to_dict(),
            'vocab': asm2vec.internal.serialization.serialize_vocabulary(self.vocab),
            'num_of_tokens': self.num_of_tokens
        }
        json.dump(d, fp)

    def populate(self, fp) -> None:
        d = json.load(fp)
        self.params = asm2vec.internal.training.Asm2VecParams(**d['params'])
        self.vocab = asm2vec.internal.serialization.deserialize_vocabulary(self.vocab)
        self.num_of_tokens = d['num_of_tokens']


class Asm2Vec:
    def __init__(self, **kwargs):
        self._params = asm2vec.internal.training.Asm2VecParams(**kwargs)
        self._repo = None

    def memento(self) -> Asm2VecMemento:
        memento = Asm2VecMemento()
        memento.params = self._params
        memento.repo = self._repo
        return memento

    def set_memento(self, memento: Asm2VecMemento) -> None:
        self._params = memento.params
        self._repo = asm2vec.internal.representative.FunctionRepository([], memento.vocab, memento.num_of_tokens)

    def train(self, funcs: List[asm2vec.asm.Function]) -> List[np.ndarray]:
        self._repo = asm2vec.internal.representative.make_function_repo(
            funcs, self._params.d, self._params.num_of_rnd_walks, self._params.jobs)

        # Calculate the frequency of each token.
        total_tokens: int = sum(map(lambda x: x.count, self._repo.vocab().values()))
        for t in self._repo.vocab().values():
            t.frequency = t.count / total_tokens

        perm = self._repo.shuffle_funcs()
        asm2vec.internal.training.train(self._repo, self._params)

        return list(map(lambda vf: vf.v, asm2vec.internal.utilities.inverse_permute(self._repo.funcs(), perm)))

    def to_vec(self, f: asm2vec.asm.Function) -> np.ndarray:
        estimate_repo = asm2vec.internal.representative.make_estimate_repo(
            self._repo, f, self._params.d, self._params.num_of_rnd_walks)
        vf = estimate_repo.funcs()[-1]

        asm2vec.internal.training.estimate(vf, estimate_repo, self._params)

        return vf.v
