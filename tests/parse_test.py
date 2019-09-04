import unittest as ut

import asm2vec.parse


test_asm = """
my_strlen:
        push    rbp
        mov     rbp, rsp
        mov     QWORD PTR [rbp-24], rdi
        mov     rax, QWORD PTR [rbp-24]
        mov     QWORD PTR [rbp-8], rax
        jmp     .L2
.L3:
        add     QWORD PTR [rbp-8], 1
.L2:
        mov     rax, QWORD PTR [rbp-8]
        movzx   eax, BYTE PTR [rax]
        test    al, al
        jne     .L3
        mov     rax, QWORD PTR [rbp-8]
        sub     rax, QWORD PTR [rbp-24]
        pop     rbp
        ret
.LC0:
        .string "%s"
.LC1:
        .string "%d\\n"
main:
        push    rbp
        mov     rbp, rsp
        add     rsp, -128
        lea     rax, [rbp-128]
        mov     rsi, rax
        mov     edi, OFFSET FLAT:.LC0
        mov     eax, 0
        call    scanf
        lea     rax, [rbp-128]
        mov     rdi, rax
        call    my_strlen
        mov     esi, eax
        mov     edi, OFFSET FLAT:.LC1
        mov     eax, 0
        call    printf
        mov     eax, 0
        leave
        ret
"""


class ParseTest(ut.TestCase):
    def test_parse_text(self):
        funcs = asm2vec.parse.parse_text(test_asm, func_names=['main', 'my_strlen'])
        self.assertEqual(2, len(funcs))
        self.assertEqual({'main', 'my_strlen'}, set(map(lambda f: f.name(), funcs)))

        funcs = dict(map(lambda f: (f.name(), f), funcs))
        main_func: asm2vec.asm.Function = funcs['main']
        my_strlen_func: asm2vec.asm.Function = funcs['my_strlen']

        self.assertListEqual(['my_strlen'], list(map(lambda f: f.name(), main_func.callees())))
        self.assertListEqual(['main'], list(map(lambda f: f.name(), my_strlen_func.callers())))
