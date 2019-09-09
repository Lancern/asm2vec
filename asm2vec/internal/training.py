from typing import *
import math
import threading
import concurrent.futures

import numpy as np

from asm2vec.asm import Instruction
from asm2vec.internal.repr import FunctionRepository
from asm2vec.internal.repr import VectorizedFunction
from asm2vec.internal.repr import Token
from asm2vec.internal.repr import VectorizedToken
from asm2vec.internal.sampling import NegativeSampler
from asm2vec.internal.atomic import LockContextManager
from asm2vec.internal.atomic import Atomic
from asm2vec.logging import asm2vec_logger


class Asm2VecParams:
    def __init__(self, **kwargs):
        self.d: int = kwargs.get('d', 200)
        self.initial_alpha: float = kwargs.get('alpha', 0.0025)
        self.alpha_update_interval: int = kwargs.get('alpha_update_interval', 10000)
        self.num_of_rnd_walks: int = kwargs.get('rnd_walks', 3)
        self.neg_samples: int = kwargs.get('neg_samples', 25)
        self.iteration: int = kwargs.get('iteration', 1)
        self.jobs: int = kwargs.get('jobs', 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'd': self.d,
            'alpha': self.initial_alpha,
            'alpha_update_interval': self.alpha_update_interval,
            'num_of_rnd_walks': self.num_of_rnd_walks,
            'neg_samples': self.neg_samples,
            'iteration': self.iteration,
            'jobs': self.jobs
        }

    def populate(self, rep: Dict[bytes, Any]) -> None:
        self.d: int = rep.get(b'd', 200)
        self.initial_alpha: float = rep.get(b'alpha', 0.0025)
        self.alpha_update_interval: int = rep.get(b'alpha_update_interval', 10000)
        self.num_of_rnd_walks: int = rep.get(b'rnd_walks', 3)
        self.neg_samples: int = rep.get(b'neg_samples', 25)
        self.iteration: int = rep.get(b'iteration', 1)
        self.jobs: int = rep.get(b'jobs', 4)


class SequenceWindow:
    def __init__(self, sequence: List[Instruction], vocabulary: Dict[str, Token]):
        self._seq = sequence
        self._vocab = vocabulary
        self._i = 1

        self._prev_ins = None
        self._curr_ins = None
        self._next_ins = None

        self._prev_ins_op = None
        self._prev_ins_args = None
        self._curr_ins_op = None
        self._curr_ins_args = None
        self._next_ins_op = None
        self._next_ins_args = None

    def move_next(self) -> bool:
        if self._i >= len(self._seq) - 1:
            return False

        def token_lookup(name) -> VectorizedToken:
            return self._vocab[name].vectorized()

        self._prev_ins = self._seq[self._i - 1]
        self._curr_ins = self._seq[self._i]
        self._next_ins = self._seq[self._i + 1]

        self._prev_ins_op = token_lookup(self._prev_ins.op())
        self._prev_ins_args = list(map(token_lookup, self._prev_ins.args()))
        self._curr_ins_op = token_lookup(self._curr_ins.op())
        self._curr_ins_args = list(map(token_lookup, self._curr_ins.args()))
        self._next_ins_op = token_lookup(self._next_ins.op())
        self._next_ins_args = list(map(token_lookup, self._next_ins.args()))

        self._i += 1

        return True

    def prev_ins(self) -> Instruction:
        return self._prev_ins

    def prev_ins_op(self) -> VectorizedToken:
        return self._prev_ins_op

    def prev_ins_args(self) -> List[VectorizedToken]:
        return self._prev_ins_args

    def curr_ins(self) -> Instruction:
        return self._curr_ins

    def curr_ins_op(self) -> VectorizedToken:
        return self._curr_ins_op

    def curr_ins_args(self) -> List[VectorizedToken]:
        return self._curr_ins_args

    def next_ins(self) -> Instruction:
        return self._next_ins

    def next_ins_op(self) -> VectorizedToken:
        return self._next_ins_op

    def next_ins_args(self) -> List[VectorizedToken]:
        return self._next_ins_args


class TrainingContext:
    class Counter:
        def __init__(self, context: 'TrainingContext', name: str, initial: int = 0):
            self._context = context
            self._name = name
            self._val = initial

        def val(self) -> int:
            with self._context.lock():
                return self._val

        def inc(self) -> int:
            with self._context.lock():
                self._val += 1
                return self._val

        def reset(self) -> int:
            with self._context.lock():
                v = self._val
                self._val = 0
                return v

    TOKENS_HANDLED_COUNTER: str = "tokens_handled"

    def __init__(self, repo: FunctionRepository, params: Asm2VecParams, is_estimating: bool = False):
        self._repo = repo
        self._params = params
        self._alpha = params.initial_alpha
        self._sampler = NegativeSampler(list(map(lambda t: (t, t.frequency), repo.vocab().values())))
        self._is_estimating = is_estimating
        self._counters = dict()
        self._lock = threading.Lock()

    def repo(self) -> FunctionRepository:
        return self._repo

    def params(self) -> Asm2VecParams:
        return self._params

    def lock(self) -> LockContextManager:
        return LockContextManager(self._lock)

    def alpha(self) -> float:
        with self.lock():
            return self._alpha

    def set_alpha(self, alpha: float) -> None:
        with self.lock():
            self._alpha = alpha

    def sampler(self) -> NegativeSampler:
        return self._sampler

    def is_estimating(self) -> bool:
        return self._is_estimating

    def create_sequence_window(self, seq: List[Instruction]) -> SequenceWindow:
        return SequenceWindow(seq, self._repo.vocab())

    def get_counter(self, name: str) -> Counter:
        with self.lock():
            return self._counters.get(name)

    def add_counter(self, name: str, initial: int = 0) -> Counter:
        with self.lock():
            c = self.__class__.Counter(self, name, initial)
            self._counters[name] = c
            return c


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(x))


def _identity(cond: bool) -> int:
    return 1 if cond else 0


def _dot_sigmoid(lhs: np.ndarray, rhs: np.ndarray) -> float:
    # noinspection PyTypeChecker
    return _sigmoid(np.dot(lhs, rhs))


def _get_inst_repr(op: VectorizedToken, args: List[VectorizedToken]) -> np.ndarray:
    if len(args) == 0:
        arg_vec = np.zeros(len(op.v))
    else:
        arg_vec = np.average(list(map(lambda tk: tk.v, args)), axis=0)
    return np.hstack((op.v, arg_vec))


def _train_vectorized(wnd: SequenceWindow, f: VectorizedFunction, context: TrainingContext) -> None:
    ct_prev = _get_inst_repr(wnd.prev_ins_op(), wnd.prev_ins_args())
    ct_next = _get_inst_repr(wnd.next_ins_op(), wnd.next_ins_args())
    delta = np.average([ct_prev, f.v, ct_next], axis=0)

    tokens = [wnd.curr_ins_op()] + wnd.curr_ins_args()

    f_grad = np.zeros(f.v.shape)
    for tk in tokens:
        # Negative sampling.
        sampled_tokens: Dict[str, VectorizedToken] = \
            dict(map(lambda x: (x.name(), x.vectorized()), context.sampler().sample(context.params().neg_samples)))
        if tk.name() not in sampled_tokens:
            sampled_tokens[tk.name()] = tk

        # The following code block tries to update the learning rate when necessary. Not required for now.
        # tokens_handled_counter = context.get_counter(TrainingContext.TOKENS_HANDLED_COUNTER)
        # if tokens_handled_counter is not None:
        #     if tokens_handled_counter.val() % context.params().alpha_update_interval == 0:
        #         # Update the learning rate.
        #         alpha = 1 - tokens_handled_counter.val() / (
        #                 context.params().iteration * context.repo().num_of_tokens() + 1)
        #         context.set_alpha(max(alpha, context.params().initial_alpha * 0.0001))

        for sp_tk in sampled_tokens.values():
            # Accumulate gradient for function vector.
            g = (_identity(tk is sp_tk) - _dot_sigmoid(delta, tk.v_pred)) * context.alpha()
            f_grad += g / 3 * tk.v_pred

            if not context.is_estimating():
                with context.lock():
                    # Update v'_t
                    tk.v_pred -= g * delta

    # Apply function gradient.
    with context.lock():
        f.v -= f_grad

    if not context.is_estimating():
        # Apply gradient to instructions.
        d = len(f_grad) // 2

        with context.lock():
            wnd.prev_ins_op().v -= f_grad[:d]
            if len(wnd.prev_ins_args()) > 0:
                prev_args_grad = f_grad[d:] / len(wnd.prev_ins_args())
                for t in wnd.prev_ins_args():
                    t.v -= prev_args_grad

            wnd.next_ins_op().v -= f_grad[:d]
            if len(wnd.next_ins_args()) > 0:
                next_args_grad = f_grad[d:] / len(wnd.next_ins_args())
                for t in wnd.next_ins_args():
                    t.v -= next_args_grad


def _train_sequence(f: VectorizedFunction, seq: List[Instruction], context: TrainingContext) -> None:
    wnd = context.create_sequence_window(seq)
    while wnd.move_next():
        _train_vectorized(wnd, f, context)


def train(repository: FunctionRepository, params: Asm2VecParams) -> None:
    context = TrainingContext(repository, params)
    context.add_counter(TrainingContext.TOKENS_HANDLED_COUNTER)

    asm2vec_logger().debug('Total number of functions: %d', len(context.repo().funcs()))
    progress = Atomic(1)

    def train_function(fn: VectorizedFunction):
        for seq in fn.sequential().sequences():
            _train_sequence(fn, seq, context)

        asm2vec_logger().debug('Function "%s" trained, progress: %f%%',
                               fn.sequential().name(), progress.value() / len(context.repo().funcs()) * 100)
        with progress.lock() as prog_proxy:
            prog_proxy.set(prog_proxy.value() + 1)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=context.params().jobs)
    futures = []
    for f in context.repo().funcs():
        futures.append(executor.submit(train_function, f))

    done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
    if len(not_done) > 0:
        raise RuntimeError('Train failed due to one or more failed task.')


def estimate(f: VectorizedFunction, estimate_repo: FunctionRepository, params: Asm2VecParams) -> np.ndarray:
    context = TrainingContext(estimate_repo, params, True)
    for seq in f.sequential().sequences():
        _train_sequence(f, seq, context)

    return f.v
