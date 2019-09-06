from typing import *

import numpy as np

import asm2vec.asm
import asm2vec.internal.util


class SequentialFunction:
    def __init__(self, f: asm2vec.asm.Function, sequences: List[List[asm2vec.asm.BasicBlock]]):
        self._f = f
        self._seq = sequences

    def func(self) -> asm2vec.asm.Function:
        return self._f

    def sequences(self) -> List[List[asm2vec.asm.BasicBlock]]:
        return self._seq


class VectorizedFunction:
    def __init__(self, f: SequentialFunction, v: np.ndarray = None, dim: int = 400):
        self._f = f
        self.v = v if v is not None else asm2vec.internal.util.make_small_ndarray(dim)

    def sequential(self) -> SequentialFunction:
        return self._f


class VectorizedToken:
    def __init__(self, name: str, v: np.ndarray = None, v_pred: np.ndarray = None, dim: int = 200):
        self._name = name
        self.v = v if v is not None else np.zeros(dim)
        self.v_pred = v_pred if v_pred is not None else asm2vec.internal.util.make_small_ndarray(dim * 2)

    def __eq__(self, other):
        if not isinstance(other, VectorizedToken):
            return False

        return self._name == other._name

    def __ne__(self, other):
        return not self.__eq__(other)

    def name(self) -> str:
        return self._name


class Token:
    def __init__(self, vt: VectorizedToken, count: int = 1):
        self._vt = vt
        self.count: int = count
        self.frequency: float = 0

    def vectorized(self) -> VectorizedToken:
        return self._vt

    def name(self) -> str:
        return self._vt.name()


class FunctionRepository:
    def __init__(self, funcs: List[VectorizedFunction], vocab: Dict[str, Token], num_of_tokens: int):
        self._funcs = funcs
        self._vocab = vocab
        self._num_of_tokens = num_of_tokens

    def funcs(self) -> List[VectorizedFunction]:
        return self._funcs

    def vocab(self) -> Dict[str, Token]:
        return self._vocab

    def num_of_tokens(self) -> int:
        return self._num_of_tokens
