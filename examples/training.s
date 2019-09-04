my_strlen_train:
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
my_strcmp_train:
        push    rbp
        mov     rbp, rsp
        mov     QWORD PTR [rbp-8], rdi
        mov     QWORD PTR [rbp-16], rsi
        jmp     .L6
.L10:
        mov     rax, QWORD PTR [rbp-8]
        movzx   edx, BYTE PTR [rax]
        mov     rax, QWORD PTR [rbp-16]
        movzx   eax, BYTE PTR [rax]
        cmp     dl, al
        je      .L7
        mov     rax, QWORD PTR [rbp-8]
        movzx   eax, BYTE PTR [rax]
        movsx   edx, al
        mov     rax, QWORD PTR [rbp-16]
        movzx   eax, BYTE PTR [rax]
        movsx   eax, al
        sub     edx, eax
        mov     eax, edx
        jmp     .L8
.L7:
        add     QWORD PTR [rbp-8], 1
        add     QWORD PTR [rbp-16], 1
.L6:
        mov     rax, QWORD PTR [rbp-8]
        movzx   eax, BYTE PTR [rax]
        test    al, al
        je      .L9
        mov     rax, QWORD PTR [rbp-16]
        movzx   eax, BYTE PTR [rax]
        test    al, al
        jne     .L10
.L9:
        mov     rax, QWORD PTR [rbp-8]
        movzx   eax, BYTE PTR [rax]
        test    al, al
        je      .L11
        mov     eax, 1
        jmp     .L8
.L11:
        mov     rax, QWORD PTR [rbp-16]
        movzx   eax, BYTE PTR [rax]
        test    al, al
        je      .L12
        mov     eax, -1
        jmp     .L8
.L12:
        mov     eax, 0
.L8:
        pop     rbp
        ret
.LC0:
        .string "%s"
.LC1:
        .string "%d\n"
main:
        push    rbp
        mov     rbp, rsp
        sub     rsp, 256
        lea     rax, [rbp-128]
        mov     rsi, rax
        mov     edi, OFFSET FLAT:.LC0
        mov     eax, 0
        call    scanf
        lea     rax, [rbp-256]
        mov     rsi, rax
        mov     edi, OFFSET FLAT:.LC0
        mov     eax, 0
        call    scanf
        lea     rax, [rbp-128]
        mov     rdi, rax
        call    my_strlen_train
        mov     esi, eax
        mov     edi, OFFSET FLAT:.LC1
        mov     eax, 0
        call    printf
        lea     rdx, [rbp-256]
        lea     rax, [rbp-128]
        mov     rsi, rdx
        mov     rdi, rax
        call    my_strcmp_train
        mov     esi, eax
        mov     edi, OFFSET FLAT:.LC1
        mov     eax, 0
        call    printf
        mov     eax, 0
        leave
        ret