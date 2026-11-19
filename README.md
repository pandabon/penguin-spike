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
| [2. An upstream test suite, bare-metal on Spike](#2-an-upstream-test-suite-bare-metal-on-spike) | XNNPACK's own unit tests on Saturn RVV, without an OS | yes |
| [3. A whole model on Spike, stage by stage](#3-a-whole-model-on-spike-stage-by-stage) | ExecuTorch on Zephyr: golden checks per stage and a per-operator profile | yes (needs the model weights from Hugging Face) |

## Quick start

```sh
git clone https://github.com/pandabon/penguin-spike.git && cd penguin-spike
scripts/setup.sh          # once: pinned submodules, Zephyr SDK (RISC-V), Spike; all inside this folder
make probe                # section 1, seconds
make xnnpack-test         # section 2, a few minutes to build, under a second to run
```

Checked from a fresh clone on Linux x86-64 in a shell with only `/usr/bin` on its PATH: setup plus both commands
take about 6 minutes and end with the probe table and `[  PASSED  ] 24 tests.`

For section 3, run `scripts/setup.sh --model` once more (a Python venv with torch and the ExecuTorch fork, plus the
board repo and Zephyr through west; several GB), then `make model`. That path (`--model`, `make model`) has not been run from a fresh clone yet; the commands it
wraps were run in our own workspace.

`scripts/setup.sh` installs nothing outside the repo: the Zephyr SDK and Spike go to `.tools/`, Python to `.venv/`,
the west workspace into the repo folder itself. Re-running it skips what is already there; `make check` reports the
state. The host needs git, curl, tar with xz, make, python3 and a C++17 compiler (to build Spike); cmake and ninja come
from pip into `.venv/`. Section 3 needs python 3.10 or newer. Linux x86-64 only for now (the SDK download is chosen for it).

| Pinned piece | Commit | What it holds |
|---|---|---|
| `zephyr-penguin-workloads` | `main` | the bare-metal XNNPACK test harness (`executorch/xnnpack-tests`), the SmolVLA Zephyr app and its export scripts (`executorch/smolvla`), the ETDump reader (`executorch/tools/etdump_inspect.py`) |
| `XNNPACK` | `penguin` | XNNPACK with Saturn RVV kernels, incl. Zvfbfa bf16 gemm and converts |
| `executorch` | `penguin-1.6` | ExecuTorch 1.6 with the Zephyr port and per-operator profiling hooks |
| `riscv-isa-sim` | upstream `609dbe0b` | the Spike these results came from (1.1.1-dev, with Zvfbfa and Zvfofp8min) |
| Zephyr SDK | 1.0.0-beta1 | GCC 14.3 + picolibc for RISC-V, and `dtc` |
| board repo + Zephyr | as pinned in `zephyr-penguin-workloads/west.yml` | fetched by west for section 3 only |

## 1. Matching Spike to the chip

**Use the chip's own ISA string.** Spike accepts Penguin's devicetree `riscv,isa` string as is:

```
rv64imafdcbv_zicsr_zifencei_zihpm_zvl256b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb_zvfofp8min_zfbfmin_zvfbfmin_zvfbfa
```

It is in the board repo's [devicetree](https://github.com/pandabon/zephyr-penguin/blob/e47fdceca142e3bda2b68ff5b0b35c25ceaaf1d1/dts/riscv/chipyard/penguin.dtsi). A hand-written string is easy to get subtly wrong. Saturn implements Zvfbfa (bf16 arithmetic selected by
`vtype.altfmt`) and Zvfbfmin, but **not** Zvfbfwma: its decoder has no `vfwmaccbf16`. A string that adds `zvfbfwma`
lets Spike run code that traps on the chip. The probe below shows it.

**Probe Saturn's bf16 / fp8 instructions.** `probes/saturn-bf16-fp8/run.sh` builds four one-instruction programs and
runs each under the chip string and under a hand-written string with `zvfbfwma`:

```sh
make probe        # or: CC=<riscv gcc> SPIKE=<spike> probes/saturn-bf16-fp8/run.sh
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
app enables it. See the board repo's [`cmake/spike.cmake`](https://github.com/pandabon/zephyr-penguin/blob/e47fdceca142e3bda2b68ff5b0b35c25ceaaf1d1/cmake/spike.cmake), [`boards/chipyard/penguin/board.cmake`](https://github.com/pandabon/zephyr-penguin/blob/e47fdceca142e3bda2b68ff5b0b35c25ceaaf1d1/boards/chipyard/penguin/board.cmake)
and the memory nodes in [`dts/riscv/chipyard/penguin.dtsi`](https://github.com/pandabon/zephyr-penguin/blob/e47fdceca142e3bda2b68ff5b0b35c25ceaaf1d1/dts/riscv/chipyard/penguin.dtsi). For a
bare-metal ELF, the same layout is:

```sh
spike --isa=<chip string> -m0x80000000:0x10000000,0x8000000:0x10000 prog.elf
```

Drop the DRAM region to check that a program really runs from the scratchpad alone.

## 2. An upstream test suite, bare-metal on Spike

**Run the library's own tests, unmodified, on the target ISA.** XNNPACK ships a GoogleTest suite per micro-kernel
family. `zephyr-penguin-workloads/executorch/xnnpack-tests` builds one suite as a bare-metal ELF and runs it on Spike
with Penguin's ISA, so every RVV kernel we add or port is checked by the upstream test before it goes near a model.

How the harness works:
- picolibc from the Zephyr SDK does start-up, `.data`/`.bss`, constructors, `malloc` and `exit`;
- a small `crt` sets `mtvec` and turns on the FP and vector units, and HTIF carries `printf` and the exit code (both
  adapted from riscv-tests and libgloss-htif, credited in the source);
- a one-thread `pthread` shim and POSIX stubs cover what GoogleTest expects from an OS;
- `run.sh` takes Spike's `--isa` from `baremetal/penguin-isa.cmake`, so the tests see the same ISA as a board build.

```sh
make xnnpack-test                 # SUITE=f32-bf16-vcvt by default; any XNNPACK test name works
# what it runs:
#   executorch/xnnpack-tests/scripts/build.sh --out build/xt f32-bf16-vcvt
#   executorch/xnnpack-tests/scripts/run.sh build/xt
```

```
[==========] Running 24 tests from 6 test suites.
[  PASSED  ] 24 tests.
ALL TESTS PASSED (rc=0)
```

Needs the Zephyr SDK 1.0.0-beta1 (its RISC-V GCC and picolibc), `cmake` and `ninja`; no Zephyr tree. A suite builds in a
few minutes and runs in under 3 s on Spike. The same suites also build as a Zephyr app (optional mode, slower:
about 16 s per run) to check the OS path.

What it caught, all on Spike:
- `f32-vapproxgelu`, 24/24, including the RVV rational kernel used for tanh-GELU.
- Our bf16 gemm micro-kernels, 68/68 and later 288/288, after each change of tile shape.
- An upstream GoogleTest issue: the old GoogleTest that XNNPACK pins has an `ASSERT_NEAR` that fails on infinities, which
  also fails upstream's scalar kernel. It was not a kernel bug.

## 3. A whole model on Spike, stage by stage

**Check every stage of a real model against PyTorch, then profile it per operator.** The SmolVLA action-expert app
(`zephyr-penguin-workloads/executorch/smolvla`) runs an ExecuTorch program (`.pte`) on Zephyr on Spike. It reads the
input tensors, runs the method, and compares the output with a golden tensor exported from PyTorch on the host. It
prints the largest absolute error, the relative L2 error and PASS or FAIL against a tolerance.

After `scripts/setup.sh --model`, `make model` exports the files and runs the app. By hand, export the model and golden
files first (from `lerobot/smolvla_base`, Apache-2.0; the weights download from Hugging Face on first use):

```sh
cd zephyr-penguin-workloads
# Python with torch, safetensors, pandas and the executorch package built from the executorch submodule
huggingface-cli download lerobot/smolvla_base      # or pass --ckpt <model.safetensors>
python executorch/smolvla/scripts/export_expert_layer.py --layer 1 --prefix-len 241 \
       --precision lerobot_bf16 --seed 0 --out OUT/expert_l1
```

Then build and run (inside the west workspace that `setup.sh --model` makes):

```sh
export EXECUTORCH_DIR=... XNNPACK_DIR=... PYTHON_EXECUTABLE=<python with flatc> FLATC_EXECUTABLE=<flatc>
export EXEC_INPUTS=OUT/expert_l1.inputs.bin EXEC_EXPECTED=OUT/expert_l1.expected.bin EXEC_TOL=2e-3 \
       ET_METHOD_POOL_SIZE=16777216 ET_TEMP_POOL_SIZE=1048576 EXEC_EMBED=hex \
       EXTRA_CMAKE="-DXNNPACK_ENABLE_RISCV_ZVFBFA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCONFIG_COMMON_LIBC_MALLOC_ARENA_SIZE=67108864"
executorch/smolvla/scripts/build.sh chipyard_penguin/penguin OUT/expert_l1.pte build/smolvla_l1
west build -d build/smolvla_l1 -t run
```

Expected (key lines):

```
execute(): minstret delta = 49009558, ...
... relL2=1.325722e-04 tol=2.000000e-03
PASS
```

About 10 minutes to build and 30 s to run. The 64 MiB `malloc` arena is required: with the app's default 1 MiB the
model fails at load with `xnn_status_out_of_memory`.

How we used it:
- **One stage at a time, then the chain.** Each stage (vision layer, language-model layer, expert layer) is exported as
  its own `.pte` with its own golden, so a mismatch points at one stage. The chain from camera images to actions was then
  checked on the host with the same `.pte` programs, because the whole chain is hours on Spike.
- **Tolerances per precision.** fp32 stages matched PyTorch at a relative L2 of about 1e-7. bf16 stages are compared
  with a tolerance (2e-3 here). The bf16-vs-fp32 difference of the original model is measured first, so the tolerance
  is not a guess.
- **Per-operator profile with ExecuTorch's own tools.** With `EXEC_ETDUMP`, the app writes an ETDump. ExecuTorch's
  Inspector and `executorch/tools/etdump_inspect.py` turn it into a table per ATen operator, with the XNNPACK operator rows
  linked back to the ATen op and Python source line. On Spike, `mcycle` counts instructions, so the table is an
  instruction profile. It ranks work; it does not predict time.
- **What the profile found.** On the first run, 61 % of the expert layer's `execute()` was a scalar 4x2 gemm: the two
  linears with n = 720 had no RVV kernel for that tile on RISC-V. Adding an RVV 7x2v kernel for that tile cut the whole action step from 9.33 G to 7.36 G instructions (-21 %), with identical output.

## Credits

- The probe's HTIF exit sequence follows the `tohost` protocol of riscv-isa-sim's front-end server.
- The instruction facts about Saturn come from [saturn-vectors](https://github.com/ucb-bar/saturn-vectors)
  (`Parameters.scala`, `Instructions.scala`, `FPFMAPipe.scala`) and the Zvfbfa draft specification
  (riscv-isa-manual pull request 2743).
