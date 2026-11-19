# Source this file (bash or zsh) to use the tools that scripts/setup.sh put inside the repo.
_ps_root=$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)
export ZEPHYR_SDK_INSTALL_DIR=$_ps_root/.tools/zephyr-sdk-1.0.0-beta1
export PATH=$_ps_root/.tools/spike/bin:$ZEPHYR_SDK_INSTALL_DIR/gnu/riscv64-zephyr-elf/bin:$ZEPHYR_SDK_INSTALL_DIR/hosttools/sysroots/x86_64-pokysdk-linux/usr/bin:$PATH
export XNNPACK_DIR=$_ps_root/XNNPACK
export EXECUTORCH_DIR=$_ps_root/executorch
[ -f "$_ps_root/.venv/bin/activate" ] && . "$_ps_root/.venv/bin/activate"
unset _ps_root
