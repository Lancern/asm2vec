from typing import *

import numpy as np

import asm2vec.asm
import asm2vec.internal.util


class SequentialFunction:
    def __init__(self, fid: int, name: str, sequences: List[List[asm2vec.asm.Instruction]]):
        self._id = fid
        self._name = name
        self._seq = sequences

    def id(self) -> int:
        return self._id

    def name(self) -> str:
        return self._name

    def sequences(self) -> List[List[asm2vec.asm.Instruction]]:
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
    def __init__(self, funcs: List[VectorizedFunction], vocab: Dict[str, Token]):
        self._funcs = funcs
        self._vocab = vocab
        self._num_of_tokens = sum(map(lambda x: x.count, vocab.values()))

    def funcs(self) -> List[VectorizedFunction]:
        return self._funcs

    def vocab(self) -> Dict[str, Token]:
        return self._vocab

    def num_of_tokens(self) -> int:
        return self._num_of_tokens


def _serialize_token(token: Token) -> Dict[str, Any]:
    return {
        'name': token.name(),
        'v': list(token.vectorized().v),
        'v_pred': list(token.vectorized().v_pred),
        'count': token.count,
        'frequency': token.frequency
    }


def _deserialize_token(rep: Dict[bytes, Any]) -> Token:
    name = rep[b'name'].decode('utf-8')
    v = np.array(rep[b'v'])
    v_pred = np.array(rep[b'v_pred'])
    count = rep[b'count']
    frequency = rep[b'frequency']

    token = Token(VectorizedToken(name, v, v_pred))
    token.count = count
    token.frequency = frequency
    return token


def serialize_vocabulary(vocab: Dict[str, Token]) -> Dict[str, Any]:
    return dict(zip(vocab.keys(), map(_serialize_token, vocab.values())))


def deserialize_vocabulary(rep: Dict[bytes, Any]) -> Dict[str, Token]:
    return dict(zip(map(lambda b: b.decode('utf-8'), rep.keys()), map(_deserialize_token, rep.values())))


def _serialize_sequence(seq: List[asm2vec.asm.Instruction]) -> List[Any]:
    return list(map(lambda instr: [instr.op(), instr.args()], seq))


def _deserialize_sequence(rep: List[Any]) -> List[asm2vec.asm.Instruction]:
    return list(map(
        lambda instr_rep: asm2vec.asm.Instruction(instr_rep[0].decode('utf-8'), instr_rep[1].decode('utf-8')), rep))


def _serialize_vectorized_function(func: VectorizedFunction, include_sequences: bool) -> Dict[str, Any]:
    data = {
        'id': func.sequential().id(),
        'name': func.sequential().name(),
        'v': list(func.v)
    }

    if include_sequences:
        data['sequences'] = list(map(_serialize_sequence, func.sequential().sequences()))

    return data


def _deserialize_vectorized_function(rep: Dict[bytes, Any]) -> VectorizedFunction:
    name = rep[b'name'].decode('utf-8')
    fid = rep[b'id']
    v = np.array(rep[b'v'])
    sequences = list(map(_deserialize_sequence, rep.get(b'sequences', [])))
    return VectorizedFunction(SequentialFunction(fid, name, sequences), v)


SERIALIZE_VOCABULARY: int = 1
SERIALIZE_FUNCTION: int = 2
SERIALIZE_FUNCTION_SEQUENCES: int = 4
SERIALIZE_ALL: int = SERIALIZE_VOCABULARY | SERIALIZE_FUNCTION | SERIALIZE_FUNCTION_SEQUENCES


def serialize_function_repo(repo: FunctionRepository, flags: int) -> Dict[str, Any]:
    data = dict()
    if (flags & SERIALIZE_VOCABULARY) != 0:
        data['vocab'] = serialize_vocabulary(repo.vocab())
    if (flags & SERIALIZE_FUNCTION) != 0:
        include_sequences = ((flags & SERIALIZE_FUNCTION_SEQUENCES) != 0)
        data['funcs'] = list(map(
            lambda f: _serialize_vectorized_function(f, include_sequences),
            repo.funcs()))

    return data


def deserialize_function_repo(rep: Dict[bytes, Any]) -> FunctionRepository:
    funcs = list(map(_deserialize_vectorized_function, rep.get(b'funcs', [])))
    vocab = deserialize_vocabulary(rep.get(b'vocab', dict()))
    return FunctionRepository(funcs, vocab)
