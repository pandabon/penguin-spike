# Each target runs one demo from the README. Run scripts/setup.sh once first; the targets source scripts/env.sh.
SHELL := /bin/bash
ENV := source scripts/env.sh &&
SUITE ?= f32-bf16-vcvt
OUT := $(CURDIR)/build

.PHONY: setup check probe xnnpack-test model-export model all

setup:                     ## submodules, Zephyr SDK (RISC-V), Spike
	scripts/setup.sh

check:
	scripts/setup.sh --check

probe:                     ## section 1: Saturn bf16/fp8 instruction probe
	$(ENV) OUT=$(OUT)/probe probes/saturn-bf16-fp8/run.sh

xnnpack-test:              ## section 2: one XNNPACK test suite, bare-metal on Spike (SUITE=...)
	mkdir -p $(OUT)
	$(ENV) cd zephyr-penguin-workloads && \
	  executorch/xnnpack-tests/scripts/build.sh --out $(OUT)/xt-$(SUITE) $(SUITE) > $(OUT)/xt-$(SUITE).build.log 2>&1 \
	  || { tail -20 $(OUT)/xt-$(SUITE).build.log; exit 1; }
	$(ENV) zephyr-penguin-workloads/executorch/xnnpack-tests/scripts/run.sh $(OUT)/xt-$(SUITE)

model-export:              ## section 3, step 1: export SmolVLA expert layer 1 + golden files (needs setup.sh --model)
	$(ENV) cd zephyr-penguin-workloads && \
	  python executorch/smolvla/scripts/export_expert_layer.py --layer 1 --prefix-len 241 \
	    --precision lerobot_bf16 --seed 0 --out $(OUT)/model/expert_l1

model: $(OUT)/model/expert_l1.pte  ## section 3, step 2: build and run it on Zephyr on Spike
	$(ENV) cd zephyr-penguin-workloads && \
	  EXEC_INPUTS=$(OUT)/model/expert_l1.inputs.bin EXEC_EXPECTED=$(OUT)/model/expert_l1.expected.bin EXEC_TOL=2e-3 \
	  ET_METHOD_POOL_SIZE=16777216 ET_TEMP_POOL_SIZE=1048576 EXEC_EMBED=hex PYTHON_EXECUTABLE=$$(command -v python) \
	  EXTRA_CMAKE="-DXNNPACK_ENABLE_RISCV_ZVFBFA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCONFIG_COMMON_LIBC_MALLOC_ARENA_SIZE=67108864" \
	  executorch/smolvla/scripts/build.sh chipyard_penguin/penguin $(OUT)/model/expert_l1.pte $(OUT)/smolvla_l1 \
	  && west build -d $(OUT)/smolvla_l1 -t run

$(OUT)/model/expert_l1.pte:
	$(MAKE) model-export

all: probe xnnpack-test
