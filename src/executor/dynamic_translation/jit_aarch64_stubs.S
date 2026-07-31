// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

    .text
    .p2align 2

    .hidden Lsx4MaterializeDeferredFlagsStaticGprJitAbi
    .global Lsx4MaterializeDeferredFlagsStaticGprJitAbi
    .type Lsx4MaterializeDeferredFlagsStaticGprJitAbi, %function
Lsx4MaterializeDeferredFlagsStaticGprJitAbi:
    bti c
    sub sp, sp, #48
    stp x4, x6, [sp, #0]
    stp x7, x8, [sp, #16]
    str x30, [sp, #32]
    bl Lsx4MaterializeDeferredFlagsBody
    ldr x30, [sp, #32]
    ldp x7, x8, [sp, #16]
    ldp x4, x6, [sp, #0]
    add sp, sp, #48
    ret
    .size Lsx4MaterializeDeferredFlagsStaticGprJitAbi, .-Lsx4MaterializeDeferredFlagsStaticGprJitAbi

    .p2align 2
    .hidden Lsx4MaterializeDeferredFlagsJitAbi
    .global Lsx4MaterializeDeferredFlagsJitAbi
    .type Lsx4MaterializeDeferredFlagsJitAbi, %function
Lsx4MaterializeDeferredFlagsJitAbi:
    bti c
    sub sp, sp, #80
    stp x1, x2, [sp, #0]
    stp x3, x4, [sp, #16]
    stp x5, x6, [sp, #32]
    stp x7, x8, [sp, #48]
    str x30, [sp, #64]
    bl Lsx4MaterializeDeferredFlagsBody
    ldr x30, [sp, #64]
    ldp x7, x8, [sp, #48]
    ldp x5, x6, [sp, #32]
    ldp x3, x4, [sp, #16]
    ldp x1, x2, [sp, #0]
    add sp, sp, #80
    ret
    .size Lsx4MaterializeDeferredFlagsJitAbi, .-Lsx4MaterializeDeferredFlagsJitAbi

    .p2align 2
    .hidden CaptureJitFaultResume
    .global CaptureJitFaultResume
    .type CaptureJitFaultResume, %function
CaptureJitFaultResume:
    bti c
    stp x19, x20, [x0, #0]
    stp x21, x22, [x0, #16]
    stp x23, x24, [x0, #32]
    stp x25, x26, [x0, #48]
    stp x27, x28, [x0, #64]
    stp x29, x30, [x0, #80]
    mov x1, sp
    str x1, [x0, #96]
    stp d8, d9, [x0, #104]
    stp d10, d11, [x0, #120]
    stp d12, d13, [x0, #136]
    stp d14, d15, [x0, #152]
    mrs x1, fpcr
    str w1, [x0, #168]
    mrs x1, fpsr
    str w1, [x0, #172]
    mov w0, wzr
    ret
    .size CaptureJitFaultResume, .-CaptureJitFaultResume

    .p2align 2
    .hidden RestoreJitFaultResume
    .global RestoreJitFaultResume
    .type RestoreJitFaultResume, %function
RestoreJitFaultResume:
    bti c
    mov x16, x0
    ldp d8, d9, [x16, #104]
    ldp d10, d11, [x16, #120]
    ldp d12, d13, [x16, #136]
    ldp d14, d15, [x16, #152]
    ldr w17, [x16, #168]
    msr fpcr, x17
    ldr w17, [x16, #172]
    msr fpsr, x17
    ldr x17, [x16, #96]
    mov sp, x17
    ldp x19, x20, [x16, #0]
    ldp x21, x22, [x16, #16]
    ldp x23, x24, [x16, #32]
    ldp x25, x26, [x16, #48]
    ldp x27, x28, [x16, #64]
    ldp x29, x30, [x16, #80]
    mov w0, #1
    ret
    .size RestoreJitFaultResume, .-RestoreJitFaultResume

    .p2align 2
    .hidden ReturnFromJitSignalContextFault
    .global ReturnFromJitSignalContextFault
    .type ReturnFromJitSignalContextFault, %function
ReturnFromJitSignalContextFault:
    bti c
    ldp d9, d10, [sp, #96]
    ldp d11, d12, [sp, #112]
    ldp d13, d14, [sp, #128]
    ldr d15, [sp, #144]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #160
    mov x0, xzr
    ret
    .size ReturnFromJitSignalContextFault, .-ReturnFromJitSignalContextFault

    .arch_extension lse
    .p2align 2
    .hidden Lsx4AtomicExchangeByteLse
    .global Lsx4AtomicExchangeByteLse
    .type Lsx4AtomicExchangeByteLse, %function
Lsx4AtomicExchangeByteLse:
    bti c
    swpalb w1, w0, [x0]
    ret
    .size Lsx4AtomicExchangeByteLse, .-Lsx4AtomicExchangeByteLse

    .macro load_repeated_half register, half
    mov \register, #\half
    movk \register, #\half, lsl #16
    movk \register, #\half, lsl #32
    movk \register, #\half, lsl #48
    .endm

    .macro compare_repeated_half register, half
    load_repeated_half x10, \half
    cmp \register, x10
    cset w10, eq
    and w9, w9, w10
    .endm

    .p2align 2
    .hidden Lsx4InvokeGeneratedCodeWithAbiCanaries
    .global Lsx4InvokeGeneratedCodeWithAbiCanaries
    .type Lsx4InvokeGeneratedCodeWithAbiCanaries, %function
Lsx4InvokeGeneratedCodeWithAbiCanaries:
    bti c
    stp x29, x30, [sp, #-112]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    str x2, [sp, #96]
    mov x16, x0
    mov x0, x1
    load_repeated_half x19, 0x1919
    load_repeated_half x20, 0x2020
    load_repeated_half x21, 0x2121
    load_repeated_half x22, 0x2222
    load_repeated_half x23, 0x2323
    load_repeated_half x24, 0x2424
    load_repeated_half x25, 0x2525
    load_repeated_half x26, 0x2626
    load_repeated_half x27, 0x2727
    load_repeated_half x28, 0x2828
    blr x16
    ldr x2, [sp, #96]
    str x0, [x2]
    mov x10, sp
    cmp x29, x10
    cset w9, eq
    compare_repeated_half x19, 0x1919
    compare_repeated_half x20, 0x2020
    compare_repeated_half x21, 0x2121
    compare_repeated_half x22, 0x2222
    compare_repeated_half x23, 0x2323
    compare_repeated_half x24, 0x2424
    compare_repeated_half x25, 0x2525
    compare_repeated_half x26, 0x2626
    compare_repeated_half x27, 0x2727
    compare_repeated_half x28, 0x2828
    mov w0, w9
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp], #112
    ret
    .size Lsx4InvokeGeneratedCodeWithAbiCanaries, .-Lsx4InvokeGeneratedCodeWithAbiCanaries

    .section .note.GNU-stack,"",%progbits
