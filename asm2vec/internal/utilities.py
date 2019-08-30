from typing import *

T = TypeVar('T')


def permute(v: List[T], permutation: List[int]) -> List[T]:
    return [v[permutation[i]] for i in range(len(permutation))]


def inverse_permute(v: List[T], permutation: List[int]) -> List[T]:
    result: List[T] = [None] * len(permutation)
    for i in range(len(permutation)):
        result[permutation[i]] = v[i]
    return result
