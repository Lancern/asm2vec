my_strlen_est:
        cmp     BYTE PTR [rdi], 0
        je      .L4
        mov     rax, rdi
.L3:
        add     rax, 1
        cmp     BYTE PTR [rax], 0
        jne     .L3
.L2:
        sub     rax, rdi
        ret
.L4:
        mov     rax, rdi
        jmp     .L2
my_strcmp_est:
        movzx   eax, BYTE PTR [rdi]
        test    al, al
        je      .L12
.L7:
        movzx   edx, BYTE PTR [rsi]
        test    dl, dl
        je      .L15
        cmp     dl, al
        jne     .L16
        add     rdi, 1
        add     rsi, 1
        movzx   eax, BYTE PTR [rdi]
        test    al, al
        jne     .L7
.L12:
        cmp     BYTE PTR [rsi], 0
        setne   dl
        movzx   edx, dl
        neg     edx
.L6:
        mov     eax, edx
        ret
.L16:
        movsx   eax, al
        movsx   edx, dl
        sub     eax, edx
        mov     edx, eax
        jmp     .L6
.L15:
        mov     edx, 1
        test    al, al
        jne     .L6
        jmp     .L12
.LC0:
        .string "%s"
.LC1:
        .string "%d\n"
main:
        sub     rsp, 264
        lea     rsi, [rsp+128]
        mov     edi, OFFSET FLAT:.LC0
        mov     eax, 0
        call    scanf
        mov     rsi, rsp
        mov     edi, OFFSET FLAT:.LC0
        mov     eax, 0
        call    scanf
        lea     rdi, [rsp+128]
        call    my_strlen_est
        mov     esi, eax
        mov     edi, OFFSET FLAT:.LC1
        mov     eax, 0
        call    printf
        mov     rsi, rsp
        lea     rdi, [rsp+128]
        call    my_strcmp_est
        mov     esi, eax
        mov     edi, OFFSET FLAT:.LC1
        mov     eax, 0
        call    printf
        mov     eax, 0
        add     rsp, 264
        ret