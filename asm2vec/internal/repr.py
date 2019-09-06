import random
from typing import *
import concurrent.futures

from asm2vec.asm import Instruction
from asm2vec.asm import BasicBlock
from asm2vec.asm import Function
from asm2vec.asm import walk_cfg
from asm2vec.repo import SequentialFunction
from asm2vec.repo import VectorizedFunction
from asm2vec.repo import VectorizedToken
from asm2vec.repo import Token
from asm2vec.repo import FunctionRepository
from asm2vec.logging import asm2vec_logger

from asm2vec.internal.atomic import Atomic


def _random_walk(f: Function) -> List[Instruction]:
    visited: Set[int] = set()
    current = f.entry()
    seq: List[Instruction] = []

    while current.id() not in visited:
        visited.add(current.id())
        for instr in current:
            seq.append(instr)
        if len(current.successors()) == 0:
            break

        current = random.choice(current.successors())

    return seq


def _edge_sampling(f: Function) -> List[List[Instruction]]:
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
        sequences.append(list(e[0]) + list(e[1]))

    return sequences


def make_sequential_function(f: Function, num_of_random_walks: int = 10) -> SequentialFunction:
    seq: List[List[Instruction]] = []

    for _ in range(num_of_random_walks):
        seq.append(_random_walk(f))

    # seq += _edge_sampling(f)

    return SequentialFunction(f.id(), f.name(), seq)


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


def _make_function_repo_helper(vocab: Dict[str, Token], funcs: List[Function],
                               dim: int, num_of_rnd_walks: int, jobs: int) -> FunctionRepository:
    progress = Atomic(1)

    vec_funcs_atomic = Atomic([])
    vocab_atomic = Atomic(vocab)

    def func_handler(f: Function):
        with vec_funcs_atomic.lock() as vfa:
            vfa.value().append(VectorizedFunction(make_sequential_function(f, num_of_rnd_walks)))

        tokens = _get_function_tokens(f, dim)
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
    done, not_done = concurrent.futures.wait(fs, return_when=concurrent.futures.FIRST_EXCEPTION)

    if len(not_done) > 0 or any(map(lambda fut: fut.cancelled() or not fut.done(), done)):
        raise RuntimeError('Not all tasks finished successfully.')

    vec_funcs = vec_funcs_atomic.value()
    repo = FunctionRepository(vec_funcs, vocab)

    # Re-calculate the frequency of each token.
    for t in repo.vocab().values():
        t.frequency = t.count / repo.num_of_tokens()

    return repo


def make_function_repo(funcs: List[Function], dim: int, num_of_rnd_walks: int, jobs: int) -> FunctionRepository:
    return _make_function_repo_helper(dict(), funcs, dim, num_of_rnd_walks, jobs)


def make_estimate_repo(vocabulary: Dict[str, Token], f: Function,
                       dim: int, num_of_rnd_walks: int) -> FunctionRepository:
    # Make a copy of the function list and vocabulary to avoid the change to affect the original trained repo.
    vocab: Dict[str, Token] = dict(**vocabulary)
    return _make_function_repo_helper(vocab, [f], dim, num_of_rnd_walks, 1)
