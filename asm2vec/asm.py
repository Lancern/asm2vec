from typing import *


class Instruction:
    def __init__(self, op: str, *args: str):
        self._op = op
        self._args = list(args)

    def op(self) -> str:
        return self._op

    def number_of_args(self) -> int:
        return len(self._args)

    def args(self) -> List[str]:
        return self._args


def parse_instruction(code: str) -> Instruction:
    sep_index = code.find(' ')
    if sep_index == -1:
        return Instruction(code)

    op = code[:sep_index]   # Operator
    args_list = list(map(str.strip, code[sep_index:].split(',')))   # Operands
    return Instruction(op, *args_list)


class BasicBlock:
    _next_unused_id: int = 1

    def __init__(self):
        # Allocate a new unique ID for the basic block.
        self._id = self.__class__._next_unused_id
        self.__class__._next_unused_id += 1

        self._instructions = []
        self._predecessors = []
        self._successors = []

    def __iter__(self):
        return self._instructions.__iter__()

    def __len__(self):
        return len(self._instructions)

    def __hash__(self):
        return self._id.__hash__()

    def __eq__(self, other):
        if not isinstance(other, BasicBlock):
            return False
        return self._id == other.id()

    def __ne__(self, other):
        return not self.__eq__(other)

    def id(self) -> int:
        return self._id

    def add_instruction(self, instr: Instruction) -> None:
        self._instructions.append(instr)

    def body_instructions(self) -> List[Instruction]:
        return self._instructions[:-1]

    def instructions(self) -> List[Instruction]:
        return self._instructions

    def add_predecessor(self, predecessor: 'BasicBlock') -> None:
        self._predecessors.append(predecessor)
        predecessor._successors.append(self)

    def add_successor(self, successor: 'BasicBlock') -> None:
        self._successors.append(successor)
        successor._predecessors.append(self)

    def first_instruction(self) -> Instruction:
        return self._instructions[0]

    def last_instruction(self) -> Instruction:
        return self._instructions[-1]

    def predecessors(self) -> List['BasicBlock']:
        return self._predecessors

    def in_degree(self) -> int:
        return len(self._predecessors)

    def successors(self) -> List['BasicBlock']:
        return self._successors

    def out_degree(self) -> int:
        return len(self._successors)


class CFGWalkerCallback:
    def __call__(self, *args, **kwargs):
        self.on_enter(*args)

    def on_enter(self, block: BasicBlock) -> None:
        pass

    def on_exit(self, block: BasicBlock) -> None:
        pass


CFGWalkerCallbackType = Union[CFGWalkerCallback, Callable[[BasicBlock], Any]]


def _walk_cfg(entry: BasicBlock, action: CFGWalkerCallbackType, visited: Set) -> None:
    if entry.id() in visited:
        return

    visited.add(entry.id())
    action(entry)

    for successor in entry.successors():
        _walk_cfg(successor, action, visited)

    if isinstance(action, CFGWalkerCallback):
        action.on_exit(entry)


def walk_cfg(entry: BasicBlock, action: CFGWalkerCallbackType) -> None:
    _walk_cfg(entry, action, set())


class Function:
    _next_unused_id = 1

    def __init__(self, entry: BasicBlock, name: str = None):
        # Allocate a unique ID for the current Function object.
        self._id = self.__class__._next_unused_id
        self.__class__._next_unused_id += 1

        self._entry = entry
        self._name = name
        self._callees = []  # Functions that are called by this function
        self._callers = []  # Functions that call this function

    def __len__(self) -> int:
        instr_count = 0

        def count_instr(block: BasicBlock) -> None:
            nonlocal instr_count
            instr_count += len(block)

        walk_cfg(self._entry, count_instr)
        return instr_count

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        if not isinstance(other, Function):
            return False
        return self._id == other.id()

    def __ne__(self, other):
        return not self.__eq__(other)

    def id(self) -> int:
        return self._id

    def entry(self) -> BasicBlock:
        return self._entry

    def name(self) -> str:
        return self._name

    def add_callee(self, f: 'Function') -> None:
        self._callees.append(f)
        f._callers.append(self)

    def callees(self) -> List['Function']:
        return self._callees

    def out_degree(self) -> int:
        return len(self._callees)

    def add_caller(self, f: 'Function') -> None:
        self._callers.append(f)
        f._callees.append(self)

    def callers(self) -> List['Function']:
        return self._callers

    def in_degree(self) -> int:
        return len(self._callers)
