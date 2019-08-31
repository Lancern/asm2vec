import unittest as ut

import asm2vec.asm as asm


class InstructionTest(ut.TestCase):
    def test_parse_instruction(self):
        ins = asm.parse_instruction('mov eax, ebx')
        self.assertEqual('mov', ins.op(), 'Operators not equal')
        self.assertListEqual(['eax', 'ebx'], ins.args(), 'Operands not equal')

    def test_parse_instruction_one_operand(self):
        ins = asm.parse_instruction('inc eax')
        self.assertEqual('inc', ins.op(), 'Operators not equal')
        self.assertListEqual(['eax'], ins.args(), 'Operands not equal')

    def test_parse_instruction_no_operands(self):
        ins = asm.parse_instruction('ret')
        self.assertEqual('ret', ins.op(), 'Operators not equal')
        self.assertListEqual([], ins.args(), 'Operands not equal')


class BasicBlockTest(ut.TestCase):
    pass


class FunctionTest(ut.TestCase):
    pass
