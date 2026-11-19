#!/usr/bin/env bash
# Build the four probes and run each on Spike under two ISA strings:
#   chip  the riscv,isa string from Penguin's devicetree (Spike accepts it as is)
#   hand  a hand-written string that adds zvfbfwma, which Saturn does not have
# Needs: a RISC-V GCC (default: the Zephyr SDK's riscv64-zephyr-elf-gcc) and spike, on PATH or as CC / SPIKE.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${OUT:-$here/build}
CC=${CC:-riscv64-zephyr-elf-gcc}
SPIKE=${SPIKE:-spike}
mkdir -p "$out"

CHIP_ISA=rv64imafdcbv_zicsr_zifencei_zihpm_zvl256b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb_zvfofp8min_zfbfmin_zvfbfmin_zvfbfa
HAND_ISA=rv64gcv_zfh_zvfh_zfbfmin_zvfbfmin_zvfbfwma_zvfbfa_zvfofp8min_zvl256b

names=([1]="Zvfbfa vfwmacc (bf16, altfmt)" [2]="Zvfbfwma vfwmaccbf16" [3]="fp8 vfwmacc (SEW 8, MX)" [4]="Zvfofp8min E4M3->bf16 cvt")

verdict() {
  case $1 in
    0) echo "runs, value ok" ;;
    12) echo "illegal instruction" ;;
    20) echo "vtype refused (vill)" ;;
    30) echo "runs, WRONG value" ;;
    124) echo "timeout" ;;
    *) echo "exit $1" ;;
  esac
}

printf '%-32s | %-22s | %-22s\n' "probe" "Spike, chip ISA" "Spike, hand ISA (+zvfbfwma)"
for c in 1 2 3 4; do
  elf=$out/probe$c.elf
  "$CC" -march=rv64gcv -mabi=lp64d -nostdlib -T "$here/link.ld" -DCASE=$c -o "$elf" "$here/probe.S"
  set +e
  timeout 10 "$SPIKE" --isa=$CHIP_ISA "$elf"; a=$?
  timeout 10 "$SPIKE" --isa=$HAND_ISA "$elf"; b=$?
  set -e
  printf '%-32s | %-22s | %-22s\n' "${names[$c]}" "$(verdict $a)" "$(verdict $b)"
done
