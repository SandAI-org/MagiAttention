# Wrapper Makefile for hopper/ffa_fa3 that adds wheel-building support.
#
# This file is copied over the submodule's hopper/Makefile during SCM builds
# (by install_flash_attn_cute.sh) so that `make install` also produces a wheel
# into MAGI_WHEEL_DIR when that variable is set.
#
# Usage (identical to the original Makefile):
#   make install ARBITRARY=1 NUM_FUNC=1,3 HDIM128=1 SM90=1
#
# With MAGI_WHEEL_DIR set, `make install` additionally writes ffa_fa3-*.whl
# into that directory.  Without MAGI_WHEEL_DIR it behaves exactly like the
# upstream Makefile.

PYTHON ?= python

# ── Build-flag variables (same defaults as upstream) ─────────────────────────
BACKWARD ?= 1
SPLIT ?= 0
PAGEDKV ?= 0
APPENDKV ?= 0
LOCAL ?= 0
SOFTCAP ?= 0
PACKGQA ?= 0
FP16 ?= 0
FP8 ?= 0
VARLEN ?= 0
CLUSTER ?= 0
HDIM64 ?= 0
HDIM96 ?= 0
HDIM128 ?= 1
HDIM192 ?= 0
HDIM256 ?= 0
SM8X ?= 1
SM90 ?= 0
VCOLMAJOR ?= 0
HDIMDIFF64 ?= 0
HDIMDIFF192 ?= 0
FORCE_UNSTABLE_API ?= 1
ARBITRARY ?= 1
NUM_FUNC ?= 3
ARBITRARY_TEST ?= 1
FLEXATTN ?= 0

# ── Flag → env-var translation (same logic as upstream) ──────────────────────
ON_VALUES := 1 true TRUE yes YES on ON
bool_false_if_on = $(if $(filter $(ON_VALUES),$($(1))),FALSE,TRUE)
bool_true_if_on  = $(if $(filter $(ON_VALUES),$($(1))),TRUE,FALSE)

export FLASH_ATTENTION_DISABLE_BACKWARD := $(call bool_false_if_on,BACKWARD)
export FLASH_ATTENTION_DISABLE_SPLIT := $(call bool_false_if_on,SPLIT)
export FLASH_ATTENTION_DISABLE_PAGEDKV := $(call bool_false_if_on,PAGEDKV)
export FLASH_ATTENTION_DISABLE_APPENDKV := $(call bool_false_if_on,APPENDKV)
export FLASH_ATTENTION_DISABLE_LOCAL := $(call bool_false_if_on,LOCAL)
export FLASH_ATTENTION_DISABLE_SOFTCAP := $(call bool_false_if_on,SOFTCAP)
export FLASH_ATTENTION_DISABLE_PACKGQA := $(call bool_false_if_on,PACKGQA)
export FLASH_ATTENTION_DISABLE_FP16 := $(call bool_false_if_on,FP16)
export FLASH_ATTENTION_DISABLE_FP8 := $(call bool_false_if_on,FP8)
export FLASH_ATTENTION_DISABLE_VARLEN := $(call bool_false_if_on,VARLEN)
export FLASH_ATTENTION_DISABLE_CLUSTER := $(call bool_false_if_on,CLUSTER)
export FLASH_ATTENTION_DISABLE_HDIM64 := $(call bool_false_if_on,HDIM64)
export FLASH_ATTENTION_DISABLE_HDIM96 := $(call bool_false_if_on,HDIM96)
export FLASH_ATTENTION_DISABLE_HDIM128 := $(call bool_false_if_on,HDIM128)
export FLASH_ATTENTION_DISABLE_HDIM192 := $(call bool_false_if_on,HDIM192)
export FLASH_ATTENTION_DISABLE_HDIM256 := $(call bool_false_if_on,HDIM256)
export FLASH_ATTENTION_DISABLE_SM80 := $(call bool_false_if_on,SM8X)
export FLASH_ATTENTION_DISABLE_SM90 := $(call bool_false_if_on,SM90)
export FLASH_ATTENTION_ENABLE_VCOLMAJOR := $(call bool_true_if_on,VCOLMAJOR)
export FLASH_ATTENTION_DISABLE_HDIMDIFF64 := $(call bool_false_if_on,HDIMDIFF64)
export FLASH_ATTENTION_DISABLE_HDIMDIFF192 := $(call bool_false_if_on,HDIMDIFF192)
export FLASH_ATTENTION_FORCE_UNSTABLE_API := $(call bool_true_if_on,FORCE_UNSTABLE_API)
export FLASH_ATTENTION_DISABLE_ARBITRARY := $(call bool_false_if_on,ARBITRARY)
export FLASH_ATTENTION_NUM_FUNC := $(NUM_FUNC)
export DISABLE_FLEX_ATTENTION := $(call bool_false_if_on,FLEXATTN)

# ── Targets ──────────────────────────────────────────────────────────────────
.PHONY: install show_flags help clean generate check_block_mask install_block_mask tt vt fm

check_block_mask:
	@$(PYTHON) -c "import create_block_mask_cuda" 2>/dev/null && echo "create_block_mask_cuda already installed" || $(MAKE) install_block_mask

install_block_mask:
	@echo "=============================================="
	@echo "Installing create_block_mask_cuda..."
	@echo "=============================================="
	cd ../csrc/utils/create_block_mask && pip install . --no-build-isolation -v
	@echo "create_block_mask_cuda installed successfully"

generate:
	@echo "Generating kernel instantiation files..."
	$(PYTHON) generate_kernels.py -o instantiations

# ── install: build wheel when MAGI_WHEEL_DIR is set, otherwise plain pip install
install: check_block_mask generate show_flags
ifdef MAGI_WHEEL_DIR
	@echo "=============================================="
	@echo "[magi-wrapper] Building ffa_fa3 wheel into $(MAGI_WHEEL_DIR)..."
	@echo "=============================================="
	$(PYTHON) -m pip wheel . --no-build-isolation --no-deps --wheel-dir "$(MAGI_WHEEL_DIR)"
	$(PYTHON) -m pip install "$(MAGI_WHEEL_DIR)"/ffa_fa3-*.whl --force-reinstall --no-deps
else
	$(PYTHON) -m pip install . --no-build-isolation -v
endif

# ── Unchanged targets from upstream ──────────────────────────────────────────
ifneq ($(and $(filter 1,$(ARBITRARY)),$(filter 1,$(ARBITRARY_TEST))),)
    TEST_FILE = test_arbitrary_mask.py
else
    TEST_FILE = test_flash_attn.py
endif

tt:
	PYTHONPATH=${PWD} python $(TEST_FILE)

vt:
	PYTHONPATH=${PWD} pytest $(TEST_FILE)::test_arbitrary_mask -v

fm:
	PYTHONPATH=${PWD} ncu --set full --nvtx --nvtx-include "flash_attn_bwd_kernel/" --nvtx-include "flash_attn_fwd_kernel/"  -f -o flash_attn.%p  python $(TEST_FILE)

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/ ffa_fa3.egg-info/ __pycache__/ .eggs/
	rm -f *.so flash_attn_cute/ffa_fa3/_C*.so
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

show_flags:
	@echo "Build flags -> env："
	@echo " BACKWARD=$(BACKWARD) -> FLASH_ATTENTION_DISABLE_BACKWARD=$(FLASH_ATTENTION_DISABLE_BACKWARD)"
	@echo " SPLIT=$(SPLIT) -> FLASH_ATTENTION_DISABLE_SPLIT=$(FLASH_ATTENTION_DISABLE_SPLIT)"
	@echo " PAGEDKV=$(PAGEDKV) -> FLASH_ATTENTION_DISABLE_PAGEDKV=$(FLASH_ATTENTION_DISABLE_PAGEDKV)"
	@echo " APPENDKV=$(APPENDKV) -> FLASH_ATTENTION_DISABLE_APPENDKV=$(FLASH_ATTENTION_DISABLE_APPENDKV)"
	@echo " LOCAL=$(LOCAL) -> FLASH_ATTENTION_DISABLE_LOCAL=$(FLASH_ATTENTION_DISABLE_LOCAL)"
	@echo " SOFTCAP=$(SOFTCAP) -> FLASH_ATTENTION_DISABLE_SOFTCAP=$(FLASH_ATTENTION_DISABLE_SOFTCAP)"
	@echo " PACKGQA=$(PACKGQA) -> FLASH_ATTENTION_DISABLE_PACKGQA=$(FLASH_ATTENTION_DISABLE_PACKGQA)"
	@echo " FP16=$(FP16) -> FLASH_ATTENTION_DISABLE_FP16=$(FLASH_ATTENTION_DISABLE_FP16)"
	@echo " FP8=$(FP8) -> FLASH_ATTENTION_DISABLE_FP8=$(FLASH_ATTENTION_DISABLE_FP8)"
	@echo " VARLEN=$(VARLEN) -> FLASH_ATTENTION_DISABLE_VARLEN=$(FLASH_ATTENTION_DISABLE_VARLEN)"
	@echo " CLUSTER=$(CLUSTER) -> FLASH_ATTENTION_DISABLE_CLUSTER=$(FLASH_ATTENTION_DISABLE_CLUSTER)"
	@echo " HDIM64=$(HDIM64) -> FLASH_ATTENTION_DISABLE_HDIM64=$(FLASH_ATTENTION_DISABLE_HDIM64)"
	@echo " HDIM96=$(HDIM96) -> FLASH_ATTENTION_DISABLE_HDIM96=$(FLASH_ATTENTION_DISABLE_HDIM96)"
	@echo " HDIM128=$(HDIM128) -> FLASH_ATTENTION_DISABLE_HDIM128=$(FLASH_ATTENTION_DISABLE_HDIM128)"
	@echo " HDIM192=$(HDIM192) -> FLASH_ATTENTION_DISABLE_HDIM192=$(FLASH_ATTENTION_DISABLE_HDIM192)"
	@echo " HDIM256=$(HDIM256) -> FLASH_ATTENTION_DISABLE_HDIM256=$(FLASH_ATTENTION_DISABLE_HDIM256)"
	@echo " SM8X=$(SM8X) -> FLASH_ATTENTION_DISABLE_SM80=$(FLASH_ATTENTION_DISABLE_SM80)"
	@echo " SM90=$(SM90) -> FLASH_ATTENTION_DISABLE_SM90=$(FLASH_ATTENTION_DISABLE_SM90)"
	@echo " VCOLMAJOR=$(VCOLMAJOR) -> FLASH_ATTENTION_ENABLE_VCOLMAJOR=$(FLASH_ATTENTION_ENABLE_VCOLMAJOR)"
	@echo " HDIMDIFF64=$(HDIMDIFF64) -> FLASH_ATTENTION_DISABLE_HDIMDIFF64=$(FLASH_ATTENTION_DISABLE_HDIMDIFF64)"
	@echo " HDIMDIFF192=$(HDIMDIFF192) -> FLASH_ATTENTION_DISABLE_HDIMDIFF192=$(FLASH_ATTENTION_DISABLE_HDIMDIFF192)"
	@echo " FORCE_UNSTABLE_API=$(FORCE_UNSTABLE_API) -> FLASH_ATTENTION_FORCE_UNSTABLE_API=$(FLASH_ATTENTION_FORCE_UNSTABLE_API)"
	@echo " ARBITRARY=$(ARBITRARY) -> FLASH_ATTENTION_DISABLE_ARBITRARY=$(FLASH_ATTENTION_DISABLE_ARBITRARY)"
	@echo " NUM_FUNC=$(NUM_FUNC) -> FLASH_ATTENTION_NUM_FUNC=$(FLASH_ATTENTION_NUM_FUNC)"
	@echo " FLEXATTN=$(FLEXATTN) -> DISABLE_FLEX_ATTENTION=$(DISABLE_FLEX_ATTENTION)"
