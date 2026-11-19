# One-instruction probes for Saturn's bf16 / fp8 vector support.
# Build one ELF per case with -DCASE=<n> (see run.sh). Each ELF exits through HTIF:
#   0            the instruction ran (and, in cases 1, 3, 4, gave the expected value)
#   10 + mcause  the instruction trapped (12 = illegal instruction)
#   20           vsetvl refused the vtype (vill set)
#   30           the instruction ran but gave a wrong value
#
# Cases:
#   1  Zvfbfa:     vfwmacc.vv with vtype.altfmt=1, SEW=16 (bf16 x bf16 + f32)
#   2  Zvfbfwma:   vfwmaccbf16.vv (Saturn does not decode it)
#   3  fp8 FMA:    vfwmacc.vv at SEW=8, altfmt=0 (E4M3 x E4M3 -> BF16; Saturn's MX FMA unit)
#   4  Zvfofp8min: vfwcvtbf16.f.f.v at SEW=8, altfmt=0 (E4M3 -> BF16)
#
# vtype bits used: vsew in [5:3] (e8=0, e16=1), vta=bit 6, vma=bit 7, altfmt=bit 8 (Zvfbfa draft spec).

#define VT_E8      0x0c0
#define VT_E16     0x0c8
#define ALTFMT     0x100

  .section .text.init
  .globl _start
_start:
  la t0, trap
  csrw mtvec, t0
  li t0, (1 << 9) | (1 << 13)        # mstatus.VS = FS = Initial
  csrs mstatus, t0
  li a1, 4                           # AVL

#if CASE == 1
  li t1, VT_E16 | ALTFMT
  vsetvl t0, a1, t1
  csrr t2, vtype
  bltz t2, vill
  li t3, 0x3f80                      # bf16 1.0
  vmv.v.x v1, t3
  vmv.v.x v2, t3
  vsetvli t0, a1, e32, m2, ta, ma
  vmv.v.i v8, 0
  vsetvl t0, a1, t1
  vfwmacc.vv v8, v1, v2              # 1.0 * 1.0 + 0 -> f32 1.0
  vsetvli t0, a1, e32, m2, ta, ma
  vmv.x.s t4, v8
  li t5, 0x3f800000
  bne t4, t5, wrong
#elif CASE == 2
  vsetivli t0, 4, e16, m1, ta, ma
  vfwmaccbf16.vv v8, v1, v2
#elif CASE == 3
  li t1, VT_E8
  vsetvl t0, a1, t1
  csrr t2, vtype
  bltz t2, vill
  li t3, 0x38                        # E4M3 1.0
  vmv.v.x v1, t3
  vmv.v.x v2, t3
  vsetvli t0, a1, e16, m2, ta, ma
  vmv.v.i v8, 0
  vsetvl t0, a1, t1
  vfwmacc.vv v8, v1, v2              # 1.0 * 1.0 + 0 -> bf16 1.0
  vsetvli t0, a1, e16, m2, ta, ma
  vmv.x.s t4, v8
  li t5, 0x3f80
  slli t4, t4, 48                    # keep the low 16 bits (no Zbb needed)
  srli t4, t4, 48
  bne t4, t5, wrong
#elif CASE == 4
  li t1, VT_E8
  vsetvl t0, a1, t1
  csrr t2, vtype
  bltz t2, vill
  li t3, 0x38                        # E4M3 1.0
  vmv.v.x v1, t3
  vfwcvtbf16.f.f.v v8, v1            # -> bf16 1.0
  vsetvli t0, a1, e16, m2, ta, ma
  vmv.x.s t4, v8
  li t5, 0x3f80
  slli t4, t4, 48                    # keep the low 16 bits (no Zbb needed)
  srli t4, t4, 48
  bne t4, t5, wrong
#else
#error "define CASE"
#endif

  li a0, 0
  j exit
vill:
  li a0, 20
  j exit
wrong:
  li a0, 30
  j exit

  .align 2
trap:
  csrr a0, mcause
  addi a0, a0, 10
exit:
  slli a0, a0, 1
  ori a0, a0, 1
  la t0, tohost
  sd a0, 0(t0)
1: j 1b

  .section .tohost, "aw"
  .align 6
  .globl tohost
tohost: .dword 0
  .align 6
  .globl fromhost
fromhost: .dword 0
