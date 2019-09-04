from typing import *
import logging

import asm2vec.asm


class AssemblySyntaxError(Exception):
    def __init__(self, message: str = None):
        self._msg = message

    def message(self) -> str:
        return self._msg


def raise_asm_syntax_error(expect: str, found: str) -> None:
    raise AssemblySyntaxError('Expect "{}", but "{}" was found.'.format(expect, found))


jmp_op = {
    'jmp', 'ja', 'jae', 'jb', 'jbe', 'jc', 'jcxz', 'jecxz', 'jrcxz', 'je', 'jg', 'jge', 'jl', 'jle', 'jna',
    'jnae', 'jnb', 'jnbe', 'jnc', 'jne', 'jng', 'jnge', 'jnl', 'jnle', 'jno', 'jnp', 'jns', 'jnz', 'jo', 'jp',
    'jpe', 'jpo', 'js', 'jz'
}

call_op = {
    'call'
}

ret_op = {
    'ret'
}

x86_64_regs = {
    'al', 'ah', 'bl', 'bh', 'cl', 'ch', 'dl', 'dh', 'spl', 'bpl', 'sil', 'dil',
    'ax', 'bx', 'cx', 'dx', 'sp', 'bp', 'si', 'di',
    'eax', 'ebx', 'ecx', 'edx', 'esp', 'ebp', 'esi', 'edi',
    'rax', 'rdx', 'rcx', 'rdx', 'rsp', 'rbp', 'rsi', 'rdi',
    'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b',
    'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w',
    'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
    'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
    'cs', 'ss', 'ds', 'es', 'fs', 'gs',
    'ecs', 'ess', 'eds', 'ees', 'efs', 'egs',
    'rcs', 'rss', 'rds', 'res', 'rfs', 'rgs'
}


def is_jmp(op: str) -> bool:
    return op.lower() in jmp_op


def is_conditional_jmp(op: str) -> bool:
    return is_jmp(op) and op.lower() != 'jmp'


def is_call(op: str) -> bool:
    return op.lower() in call_op


def is_ret(op: str) -> bool:
    return op.lower() in ret_op


def is_reg(arg: str) -> bool:
    return arg.lower() in x86_64_regs


class CFGBuilder:
    def __init__(self, context: 'ParseContext'):
        self._context = context
        self._blocks: List[asm2vec.asm.BasicBlock] = []
        self._active_block = -1
        self._block_labels: Dict[str, int] = dict()

    def _logger(self) -> logging.Logger:
        return self._context.logger().getChild(self.__class__.__name__)

    def _allocate_block(self) -> int:
        self._blocks.append(asm2vec.asm.BasicBlock())
        return len(self._blocks) - 1

    def _allocate_named_block(self, name: str) -> int:
        if name in self._block_labels:
            return self._block_labels[name]
        else:
            idx = self._allocate_block()
            self._block_labels[name] = idx
            return idx

    def _get_active_block(self) -> asm2vec.asm.BasicBlock:
        return self._blocks[self._active_block]

    def _set_active_block(self, block_id: int) -> None:
        self._active_block = block_id

    def _has_active_block(self) -> bool:
        return self._active_block != -1

    def _close_active_block(self) -> None:
        self._active_block = -1

    def _add_jmp(self, op: str, args: List[str]) -> None:
        if len(args) != 1:
            raise_asm_syntax_error('Jump with single operand', '{} operands'.format(len(args)))
        cur_block = self._get_active_block()
        self._close_active_block()
        if is_conditional_jmp(op):
            # Allocate another basic block for more instructions since the current code point is reachable.
            # This may produce some empty basic blocks in the final output.
            self._set_active_block(self._allocate_block())
            self._get_active_block().add_predecessor(cur_block)

    def add_instr(self, op: str, args: List[str]) -> None:
        if not self._has_active_block():
            # Allocate a new basic block.
            self._set_active_block(self._allocate_block())

        self._get_active_block().add_instruction(asm2vec.asm.Instruction(op, *args))
        if is_jmp(op):
            self._add_jmp(op, args)
        elif is_ret(op):
            # `ret` instruction encountered. Close current active block.
            self._close_active_block()

    def set_label(self, label: str) -> None:
        block_id = self._block_labels.get(label, -1)
        if block_id == -1:
            # Test if the current active block is empty in which case we can reuse it.
            if self._has_active_block() and len(self._get_active_block()) == 0:
                self._block_labels[label] = self._active_block
            else:
                # Open a new block for the label.
                block_id = self._allocate_block()
                self._block_labels[label] = block_id
                # Link the new block with the previously-active block.
                if self._has_active_block():
                    self._get_active_block().add_successor(self._blocks[block_id])
                self._set_active_block(block_id)
        else:
            self._set_active_block(block_id)

    def build(self) -> List[asm2vec.asm.Function]:
        func_entries: Dict[str, int] = dict()

        # Walk through all instructions and fix block relations formed by jump and call instructions.
        for blk in self._blocks:
            for inst in blk:
                if is_jmp(inst.op()):
                    target = inst.args()[0]
                    if target in self._block_labels:
                        blk.add_successor(self._blocks[self._block_labels[target]])
                elif is_call(inst.op()):
                    target = inst.args()[0]
                    if target in self._block_labels and target not in func_entries:
                        func_entries[target] = self._block_labels[target]

        for func_name in self._context.options().func_names():
            if func_name not in self._block_labels:
                self._logger().warning('Cannot find function "{}"', func_name)
                continue
            if func_name not in func_entries:
                func_entries[func_name] = self._block_labels[func_name]

        funcs: Dict[str, asm2vec.asm.Function] = \
            dict(map(lambda x: (x[0], asm2vec.asm.Function(self._blocks[x[1]], x[0])), func_entries.items()))

        # Fix function call relation.
        for (name, f) in funcs.items():
            def block_action(block: asm2vec.asm.BasicBlock) -> None:
                for instr in block:
                    if is_call(instr.op()):
                        callee_name = instr.args()[0]
                        if callee_name in funcs:
                            f.add_callee(funcs[callee_name])

            asm2vec.asm.walk_cfg(f.entry(), block_action)

        # TODO: Implement Selective Callee Expansion here.

        return list(funcs.values())


class ParseOptions:
    def __init__(self, **kwargs):
        self._func_names = kwargs.get('func_names', [])

    def func_names(self) -> List[str]:
        return self._func_names


class ParseContext:
    def __init__(self, **kwargs):
        self._builder = CFGBuilder(self)
        self._options = ParseOptions(**kwargs)
        self._logger = logging.getLogger('asm2vec.ParseContext')

    def logger(self) -> logging.Logger:
        return self._logger

    def options(self) -> ParseOptions:
        return self._options

    def builder(self) -> CFGBuilder:
        return self._builder


'''

Parser rules for input assembly file:

program
    : asm_line*
    ;

asm_line
    : asm_label '\n'
    | BLANKS asm_instr '\n'
    ;

asm_label
    : ASM_LABEL_ID ':'
    ;

asm_instr
    : ASM_INSTR_OP ' ' asm_instr_arg_list
    ;

asm_instr_arg_list
    : ASM_INSTR_ARG (',' asm_instr_arg_list)?
    | /* epsilon */
    ;

BLANKS : [ \n\t]+;

'''


def is_fullmatch(pattern, s: str) -> bool:
    return pattern.fullmatch(s) is not None


def parse_asm_label(ln: str, context: ParseContext) -> None:
    stripped = ln.strip()
    if stripped[-1] != ':':
        raise_asm_syntax_error('asm_label', ln)

    context.builder().set_label(stripped[:-1])


def parse_asm_instr(ln: str, context: ParseContext) -> None:
    delim_index = ln.find(' ')
    args = []
    if delim_index == -1:
        op = ln
    else:
        op = ln[:delim_index]
        args = list(map(lambda arg: arg.strip(), ln[delim_index + 1:].split(',')))

    context.builder().add_instr(op, args)


def parse_asm_line(ln: str, context: ParseContext) -> None:
    if len(ln.strip()) == 0:
        return

    if ln[0].isspace():
        # Expect production asm_line -> BLANKS asm_instr '\n'
        parse_asm_instr(ln.strip(), context)
    else:
        # Expect production asm_line -> asm_label
        parse_asm_label(ln, context)


def parse_asm_lines(lines: Iterable[str], **kwargs) -> List[asm2vec.asm.Function]:
    context = ParseContext(**kwargs)
    for ln in lines:
        parse_asm_line(ln, context)
    return context.builder().build()
