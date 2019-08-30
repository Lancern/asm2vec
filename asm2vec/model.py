from typing import *

import numpy as np

import asm2vec.internal.training as training
from asm2vec.asm import Function
from asm2vec.internal.representative import make_function_repo
from asm2vec.internal.representative import make_estimate_repo
from asm2vec.internal.utilities import inverse_permute


class Asm2Vec:
    def __init__(self, **kwargs):
        self._params = training.Asm2VecParams(**kwargs)
        self._repo = None

    def train(self, funcs: List[Function]) -> List[np.ndarray]:
        self._repo = make_function_repo(funcs, self._params.d, self._params.num_of_rnd_walks)

        # Calculate the frequency of each token.
        total_tokens: int = sum(map(lambda x: x.count, self._repo.vocab().values()))
        for t in self._repo.vocab().values():
            t.frequency = t.count / total_tokens

        perm = self._repo.shuffle_funcs()
        training.train(self._repo, self._params)

        return list(map(lambda vf: vf.v, inverse_permute(self._repo.funcs(), perm)))

    def to_vec(self, f: Function) -> np.ndarray:
        estimate_repo = make_estimate_repo(self._repo, f, self._params.d, self._params.num_of_rnd_walks)
        vf = estimate_repo.funcs()[-1]

        training.estimate(vf, estimate_repo, self._params)

        return vf.v
