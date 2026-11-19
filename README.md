# penguin-spike

How we use [Spike](https://github.com/riscv-software-src/riscv-isa-sim) to check software for the Penguin SoC
(Shuttle + Saturn vector unit + the Atlas accelerator) before it reaches FireSim, the FPGA or the chip. This is a
reference: each section shows one technique, the exact commands, what it caught, and where the full code lives. The
code itself stays in its own repo, pinned here as a submodule at the commit the results came from.

Spike is a functional model. Nothing here says anything about timing, caches or bus behaviour, and every claim below says
whether it ran on Spike, on RTL simulation or on the chip.

| Section | What it shows | Runnable here |
|---|---|---|
| [1. Matching Spike to the chip](#1-matching-spike-to-the-chip) | the chip's ISA string and memory map on Spike; a probe of Saturn's bf16/fp8 instructions | `probes/saturn-bf16-fp8/run.sh` |
| [2. An upstream test suite, bare-metal on Spike](#2-an-upstream-test-suite-bare-metal-on-spike) | XNNPACK's own unit tests on Saturn RVV, without an OS | TODO |
| [3. A whole model on Spike, stage by stage](#3-a-whole-model-on-spike-stage-by-stage) | ExecuTorch on Zephyr: golden checks per stage and a per-operator profile | TODO |
| [4. Modelling Atlas inside Spike](#4-modelling-atlas-inside-spike) | a Spike plugin for Atlas's host interface and scalar core | description only |

## Setup

```sh
git clone --recursive https://github.com/pandabon/penguin-spike.git
```

You need `spike` (riscv-isa-sim with Zvfbfa and Zvfofp8min, upstream since 2025) and an LLVM 18 or newer `clang` + `lld`
with the RISC-V target. Sections 2 and 3 list their own extra tools.

## 1. Matching Spike to the chip

**Use the chip's own ISA string.** Spike accepts Penguin's devicetree `riscv,isa` string as is:

```
rv64imafdcbv_zicsr_zifencei_zihpm_zvl256b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb_zvfofp8min_zfbfmin_zvfbfmin_zvfbfa
```

A hand-written string is easy to get subtly wrong. Saturn implements Zvfbfa (bf16 arithmetic selected by
`vtype.altfmt`) and Zvfbfmin, but **not** Zvfbfwma: its decoder has no `vfwmaccbf16`. A string that adds `zvfbfwma`
lets Spike run code that traps on the chip. The probe below shows it.

**Probe Saturn's bf16 / fp8 instructions.** `probes/saturn-bf16-fp8/run.sh` builds four one-instruction programs and
runs each under the chip string and under a hand-written string with `zvfbfwma`:

```sh
CLANG=/path/to/clang SPIKE=/path/to/spike probes/saturn-bf16-fp8/run.sh
```

| Probe | Spike, chip ISA | Spike, + zvfbfwma | Saturn (from its RTL) |
|---|---|---|---|
| Zvfbfa `vfwmacc.vv`, bf16 via `altfmt` | runs, value ok | runs, value ok | implemented |
| Zvfbfwma `vfwmaccbf16.vv` | illegal instruction | runs | **not decoded** |
| fp8 `vfwmacc.vv` at SEW 8 (Saturn's MX FMA) | illegal instruction | illegal instruction | implemented (passed Saturn's `vec-mx-fmacc` on RTL simulation) |
| Zvfofp8min E4M3 → bf16 `vfwcvtbf16.f.f.v` | runs, value ok | runs, value ok | implemented |

So on Spike, a bf16 path built on Zvfbfa and the fp8 conversions can be checked bit for bit. An fp8 multiply-add path
cannot: Spike has no model of Saturn's MX FMA, so it needs RTL simulation, FireSim or the chip. Saturn's own MX
benchmarks show the same split (the ones using fp8 `vfwmacc` trap on Spike, the rest pass).

**Give Spike the on-chip memories.** The board port derives the ISA string from Kconfig and passes `-m` regions taken
from the devicetree: DRAM at 0x8000_0000, the 64 KiB scratchpad at 0x0800_0000, and Atlas's VMEM at 0x2000_0000 when an
app enables it. See `zephyr-penguin/cmake/spike.cmake` and `zephyr-penguin/boards/chipyard/penguin/board.cmake`. For a
bare-metal ELF, the same layout is:

```sh
spike --isa=<chip string> -m0x80000000:0x10000000,0x8000000:0x10000 prog.elf
```

Drop the DRAM region to check that a program really runs from the scratchpad alone.

## 2. An upstream test suite, bare-metal on Spike

TODO (waiting for the harness to be published).

## 3. A whole model on Spike, stage by stage

TODO (waiting for the app to be published).

## 4. Modelling Atlas inside Spike

We have a prototype Spike plugin for Atlas. It models the host-visible windows (instruction memory at 0x2_0000, CSRs at
0x4_0000, VMEM at 0x2000_0000) and Atlas's scalar core, so a driver and its host code run on Spike. It does not model
DMA, the matrix or vector units, or numerics, so a real kernel stops at its first DMA. It is not public yet. The next step
we have in mind is to use the Atlas team's [npu_model](https://github.com/ucb-ee194-tapeout/npu_model) as the engine
behind the same windows, so one Spike run checks the host side and Atlas's results together.

Things about Spike's plugin interface we learned the hard way (Spike 1.1.1-dev):
- Build the plugin as a shared object and do **not** link `libriscv`. The `spike` binary already exports it; a second
  copy registers devices into a separate map, and Spike then reports "Plugin not found".
- Load it with `--extlib=lib.so --device=<name>,<args>`. Each device instance has one base address, so Atlas's three
  windows take three `--device` instances that share one model object.
- Plugin devices advance only in `tick()`, which Spike calls once every 5,000 instructions. Let the model also step on
  host reads of its registers, or poll loops crawl.
- Plugin devices never appear in Spike's generated devicetree; a static board devicetree is needed.
- `parse()` receives the `sim_t*`, whose `addr_to_mem()` gives direct access to RAM, which is enough for a DMA model.

## Credits

- The probe's HTIF exit sequence follows the `tohost` protocol of riscv-isa-sim's front-end server.
- The instruction facts about Saturn come from [saturn-vectors](https://github.com/ucb-bar/saturn-vectors)
  (`Parameters.scala`, `Instructions.scala`, `FPFMAPipe.scala`) and the Zvfbfa draft specification
  (riscv-isa-manual pull request 2743).
