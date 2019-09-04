from typing import *
import random

T = TypeVar('T')


class NegativeSampler:
    def __init__(self, distribution: List[Tuple[T, float]], alpha: float = 3 / 4):
        self._values = list(map(lambda x: x[0], distribution))
        self._weights = list(map(lambda x: x[1] ** alpha, distribution))

    def sample(self, k: int) -> List[T]:
        return random.choices(self._values, self._weights, k=k)
