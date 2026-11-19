#!/usr/bin/env bash
# One-time setup. Everything goes inside this repo (.tools/, .venv/, the west workspace folders); nothing is
# installed system-wide, and every step is skipped when its result already exists.
#
#   scripts/setup.sh            submodules + Zephyr SDK (RISC-V only) + Spike: enough for sections 1 and 2
#   scripts/setup.sh --model    also the west workspace (board repo + Zephyr) and a Python venv for section 3
#   scripts/setup.sh --check    only report what is there
#
# Host needs: git, curl, tar, xz, make, a C++17 compiler (for Spike) and python3 (>= 3.10 for --model).
# cmake and ninja come from pip into .venv, so the host's versions do not matter.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
tools=$root/.tools
SDK_VER=1.0.0-beta1
SDK=$tools/zephyr-sdk-$SDK_VER
SDK_URL=https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v$SDK_VER
mode=${1:-}

say() { printf '\n== %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

check() {
  local ok=1
  printf '%-12s %s\n' tool status
  for t in git curl tar make g++ python3; do
    if have $t; then printf '%-12s ok\n' $t; else printf '%-12s MISSING\n' $t; ok=0; fi
  done
  for t in cmake ninja; do
    if [ -x .venv/bin/$t ]; then printf '%-12s ok (.venv)\n' $t; else printf '%-12s not set up\n' $t; ok=0; fi
  done
  [ -x "$SDK/gnu/riscv64-zephyr-elf/bin/riscv64-zephyr-elf-gcc" ] && printf '%-12s ok (%s)\n' sdk "$SDK" || { printf '%-12s not set up\n' sdk; ok=0; }
  [ -x "$tools/spike/bin/spike" ] && printf '%-12s ok\n' spike || { printf '%-12s not built\n' spike; ok=0; }
  [ -d .west ] && printf '%-12s ok\n' west-ws || printf '%-12s not set up (only section 3 needs it: --model)\n' west-ws
  [ -x .venv/bin/west ] && printf '%-12s ok\n' model-py || printf '%-12s not set up (only section 3 needs it: --model)\n' model-py
  [ $ok = 1 ]
}

if [ "$mode" = --check ]; then check; exit $?; fi

say "submodules (only the commits we pin, shallow)"
git submodule update --init zephyr-penguin-workloads
git submodule update --init --depth 1 XNNPACK executorch riscv-isa-sim
# The XNNPACK test harness also builds GoogleTest and XNNPACK's helper libraries from ExecuTorch's tree.
git -C executorch submodule update --init --depth 1 third-party/googletest \
  backends/xnnpack/third-party/{pthreadpool,cpuinfo,FXdiv,FP16}

say "Python venv (.venv) with cmake and ninja"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip cmake ninja

say "Zephyr SDK $SDK_VER, RISC-V toolchain and host tools (dtc), into .tools/"
if [ ! -x "$SDK/gnu/riscv64-zephyr-elf/bin/riscv64-zephyr-elf-gcc" ]; then
  mkdir -p "$tools"
  curl -fL "$SDK_URL/zephyr-sdk-${SDK_VER}_linux-x86_64_minimal.tar.xz" | tar -xJ -C "$tools"
  # -t: the RISC-V GNU toolchain only; -h: host tools (dtc for Zephyr and Spike). No -c: nothing is registered
  # in ~/.cmake; scripts/env.sh sets ZEPHYR_SDK_INSTALL_DIR instead.
  (cd "$SDK" && ./setup.sh -t riscv64-zephyr-elf -h)
fi

say "Spike (riscv-isa-sim at the pinned commit), into .tools/spike"
if [ ! -x "$tools/spike/bin/spike" ]; then
  # Spike needs dtc at run time and at build time; the SDK's host tools provide it.
  # shellcheck disable=SC1091
  source "$root/scripts/env.sh"
  mkdir -p "$tools/spike-build"
  (cd "$tools/spike-build" && "$root/riscv-isa-sim/configure" --prefix="$tools/spike" >/dev/null &&
     make -j"${JOBS:-8}" >/dev/null && make install >/dev/null)
fi

if [ "$mode" = --model ]; then
  say "Python packages for section 3: west, Zephyr's build requirements, torch and the ExecuTorch fork"
  .venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
    { echo "section 3 needs python >= 3.10 in .venv: remove .venv and rerun with a newer python3 first on PATH"; exit 1; }
  .venv/bin/pip install -q west
  say "west workspace: the board repo and the Zephyr fork it pins, as listed in zephyr-penguin-workloads/west.yml"
  [ -d .west ] || .venv/bin/west init -l zephyr-penguin-workloads
  .venv/bin/west update
  .venv/bin/pip install -q -r zephyr/scripts/requirements-base.txt
  .venv/bin/pip install -q torch safetensors pandas huggingface_hub
  # ExecuTorch's Python package (exporter + flatc) built from the pinned fork; it needs the fork's own submodules.
  git -C executorch submodule update --init --recursive --depth 1
  (source .venv/bin/activate && cd executorch && ./install_executorch.sh)
fi

say "status"
check || true
echo
echo "Next: source scripts/env.sh, then make probe / make xnnpack-test (and, after --model, make model)."
