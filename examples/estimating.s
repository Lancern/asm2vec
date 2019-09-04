my_strlen_train:
        cmp     BYTE PTR [rdi], 0
        je      .L4
        mov     rax, rdi
.L3:
        add     rax, 1
        cmp     BYTE PTR [rax], 0
        jne     .L3
        sub     eax, edi
        ret
.L4:
        xor     eax, eax
        ret
my_strcmp_train:
        jmp     .L17
.L20:
        movsx   edx, BYTE PTR [rsi]
        test    dl, dl
        je      .L18
        cmp     al, dl
        jne     .L19
        add     rdi, 1
        add     rsi, 1
.L17:
        movsx   eax, BYTE PTR [rdi]
        test    al, al
        jne     .L20
        xor     eax, eax
        cmp     BYTE PTR [rsi], 0
        setne   al
        neg     eax
        ret
.L18:
        mov     eax, 1
        ret
.L19:
        sub     eax, edx
        ret
.LC0:
        .string "%s"
.LC1:
        .string "%d\n"
main:
        push    rbx
        mov     edi, OFFSET FLAT:.LC0
        xor     eax, eax
        sub     rsp, 256
        mov     rbx, rsp
        mov     rsi, rbx
        call    scanf
        lea     rsi, [rsp+128]
        xor     eax, eax
        mov     edi, OFFSET FLAT:.LC0
        call    scanf
        cmp     BYTE PTR [rsp], 0
        mov     rsi, rbx
        je      .L22
.L23:
        add     rsi, 1
        cmp     BYTE PTR [rsi], 0
        jne     .L23
.L22:
        sub     rsi, rbx
        mov     edi, OFFSET FLAT:.LC1
        xor     eax, eax
        call    printf
        movzx   eax, BYTE PTR [rsp]
        lea     rcx, [rsp+128]
        mov     rdx, rbx
        test    al, al
        jne     .L24
        jmp     .L25
.L28:
        cmp     dil, al
        jne     .L38
        movzx   eax, BYTE PTR [rdx+1]
        add     rdx, 1
        add     rcx, 1
        test    al, al
        je      .L25
.L24:
        movsx   edi, BYTE PTR [rcx]
        test    dil, dil
        jne     .L28
        mov     esi, 1
.L27:
        mov     edi, OFFSET FLAT:.LC1
        xor     eax, eax
        call    printf
        add     rsp, 256
        xor     eax, eax
        pop     rbx
        ret
.L25:
        xor     esi, esi
        cmp     BYTE PTR [rcx], 0
        setne   sil
        neg     esi
        jmp     .L27
.L38:
        movsx   esi, al
        sub     esi, edi
        jmp     .L27