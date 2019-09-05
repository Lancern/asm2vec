from typing import *

import numpy as np

from asm2vec.internal.representative import VectorizedToken
from asm2vec.internal.representative import Token
from asm2vec.internal.representative import FunctionRepository


def serialize_ndarray(v: np.ndarray) -> List[float]:
    return list(v)


def deserialize_ndarray(rep: List[float]) -> np.ndarray:
    return np.array(rep)


def serialize_token(token: Token) -> Dict[str, Any]:
    return {
        'name': token.name(),
        'v': serialize_ndarray(token.vectorized().v),
        'v_pred': serialize_ndarray(token.vectorized().v_pred),
        'count': token.count,
        'frequency': token.frequency
    }


def deserialize_token(rep: Dict[str, Any]) -> Token:
    name = rep['name']
    v = rep['v']
    v_pred = rep['v_pred']
    count = rep['count']
    frequency = rep['frequency']

    token = Token(VectorizedToken(name, v, v_pred))
    token.count = count
    token.frequency = frequency
    return token


def serialize_vocabulary(vocab: Dict[str, Token]) -> Dict[str, Any]:
    return dict(zip(vocab.keys(), map(serialize_token, vocab.values())))


def deserialize_vocabulary(rep: Dict[str, Any]) -> Dict[str, Token]:
    return dict(zip(rep.keys(), map(deserialize_token, rep.values())))


def serialize_repo(repo: FunctionRepository) -> Dict[str, Any]:
    return {
        'vocab': serialize_vocabulary(repo.vocab()),
        'num_of_tokens': repo.num_of_tokens()
    }


def deserialize_repo(rep: Dict[str, Any]) -> FunctionRepository:
    funcs = rep['funcs']
    vocab = rep['vocab']
    num_of_tokens = rep['num_of_tokens']
    return FunctionRepository(funcs, vocab, num_of_tokens)
