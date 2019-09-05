import random
from typing import *
import concurrent.futures

import numpy as np

from asm2vec.asm import Instruction
from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import walk_cfg
from asm2vec.internal.utilities import permute
from asm2vec.internal.atomic import Atomic
from asm2vec.logging import asm2vec_logger


def _make_small_ndarray(dim: int) -> np.ndarray:
    rng = np.random.default_rng()
    return (rng.random(dim) - 0.5) / dim


class SequentialFunction:
    def __init__(self, f: Function, sequences: List[List[BasicBlock]]):
        self._f = f
        self._seq = sequences

    def func(self) -> Function:
        return self._f

    def sequences(self) -> List[List[BasicBlock]]:
        return self._seq


def flat_sequence(seq: List[BasicBlock]) -> List[Instruction]:
    instr_seq = []
    for block in seq:
        instr_seq += list(block)
    return instr_seq


class VectorizedFunction:
    def __init__(self, f: SequentialFunction, v: np.ndarray = None, dim: int = 400):
        self._f = f
        self.v = v if v is not None else _make_small_ndarray(dim)

    def sequential(self) -> SequentialFunction:
        return self._f


class VectorizedToken:
    def __init__(self, name: str, v: np.ndarray = None, v_pred: np.ndarray = None, dim: int = 200):
        self._name = name
        self.v = v if v is not None else np.zeros(dim)
        self.v_pred = v_pred if v_pred is not None else _make_small_ndarray(dim * 2)

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

    def shuffle_funcs(self) -> List[int]:
        perm = list(range(len(self._funcs)))
        self._funcs = permute(self._funcs, perm)
        return perm

    def vocab(self) -> Dict[str, Token]:
        return self._vocab

    def num_of_tokens(self) -> int:
        return self._num_of_tokens


def _random_walk(f: Function) -> List[BasicBlock]:
    visited: Set[int] = set()
    current = f.entry()
    block_seq: List[BasicBlock] = []

    while current.id() not in visited:
        visited.add(current.id())
        block_seq.append(current)
        if len(current.successors()) == 0:
            break

        current = random.choice(current.successors())

    return block_seq


def _edge_sampling(f: Function) -> List[List[BasicBlock]]:
    edges: List[Tuple[BasicBlock, BasicBlock]] = []

    def collect_edges(block: BasicBlock) -> None:
        nonlocal edges
        for successor in block.successors():
            edges.append((block, successor))

    walk_cfg(f.entry(), collect_edges)

    visited_edges: Set[Tuple[int, int]] = set()
    sequences = []
    while len(visited_edges) < len(edges):
        e = random.choice(edges)
        visited_edges.add((e[0].id(), e[1].id()))
        sequences.append([e[0], e[1]])

    return sequences


def make_sequential_function(f: Function, num_of_random_walks: int = 10) -> SequentialFunction:
    seq: List[List[BasicBlock]] = []

    for _ in range(num_of_random_walks):
        seq.append(_random_walk(f))

    seq += _edge_sampling(f)

    return SequentialFunction(f, seq)


def _get_function_tokens(f: Function, dim: int = 200) -> List[VectorizedToken]:
    tokens: List[VectorizedToken] = []

    def collect_tokens(block: BasicBlock) -> None:
        nonlocal tokens
        for ins in block:
            tk: List[str] = [ins.op()] + ins.args()
            for t in tk:
                tokens.append(VectorizedToken(t, None, None, dim))

    walk_cfg(f.entry(), collect_tokens)
    return tokens


def _make_function_repo_helper(vec_funcs: List[VectorizedFunction], vocab: Dict[str, Token], num_of_tokens: int,
                               funcs: List[Function], dim: int, num_of_rnd_walks: int, jobs: int) -> FunctionRepository:
    progress = Atomic(1)

    vec_funcs_atomic = Atomic(vec_funcs)
    vocab_atomic = Atomic(vocab)
    num_of_tokens_atomic = Atomic(num_of_tokens)

    def func_handler(f: Function):
        with vec_funcs_atomic.lock() as vfa:
            vfa.value().append(VectorizedFunction(make_sequential_function(f, num_of_rnd_walks)))

        tokens = _get_function_tokens(f, dim)
        with num_of_tokens_atomic.lock() as nta:
            nta.set(nta.value() + len(tokens))

        for tk in tokens:
            with vocab_atomic.lock() as va:
                if tk.name() in va.value():
                    va.value()[tk.name()].count += 1
                else:
                    va.value()[tk.name()] = Token(tk)

        asm2vec_logger().debug('Sequence generated for function "%s", progress: %f%%',
                               f.name(), progress.value() / len(funcs) * 100)
        with progress.lock() as prog:
            prog.set(prog.value() + 1)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
    fs = []
    for fn in funcs:
        fs.append(executor.submit(func_handler, fn))
    concurrent.futures.wait(fs)

    return FunctionRepository(vec_funcs, vocab, num_of_tokens)


def make_function_repo(funcs: List[Function], dim: int, num_of_rnd_walks: int, jobs: int) -> FunctionRepository:
    return _make_function_repo_helper([], dict(), 0, funcs, dim, num_of_rnd_walks, jobs)


def make_estimate_repo(trained_repo: FunctionRepository, f: Function,
                       dim: int, num_of_rnd_walks: int) -> FunctionRepository:
    # Make a copy of the function list and vocabulary to avoid the change to affect the original trained repo.
    vec_funcs: List[VectorizedFunction] = list(trained_repo.funcs())
    vocab: Dict[str, Token] = dict(**trained_repo.vocab())
    num_of_tokens: int = trained_repo.num_of_tokens()

    return _make_function_repo_helper(vec_funcs, vocab, num_of_tokens, [f], dim, num_of_rnd_walks, 1)
