from typing import *

from asm2vec.asm import Instruction
from asm2vec.internal.representative import Token


InstructionSequence = List[Instruction]

Vocabulary = Dict[str, Token]
