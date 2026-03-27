#!/usr/bin/env python3
"""Patch create_block_mask/setup.py for headless (no-GPU) build environments.

The submodule's setup.py unconditionally calls torch.cuda.is_available() to
detect the target GPU architecture. On SCM build machines there is no GPU, so
this fails. This script patches the function to fall back to the
MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY environment variable.

Usage:
    python3 scripts/patch_create_block_mask.py <path-to-setup.py>
"""

import sys
import pathlib

OLD = """\
def get_cuda_gencode_flags():
    \"\"\"Detect current GPU architecture and return appropriate -gencode flags.\"\"\"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot determine target architecture")
    capability = torch.cuda.get_device_capability()
    arch = capability[0] * 10 + capability[1]
    return ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]"""

NEW = """\
def get_cuda_gencode_flags():
    \"\"\"Detect current GPU architecture and return appropriate -gencode flags.\"\"\"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        arch = capability[0] * 10 + capability[1]
        return ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
    cc_env = os.environ.get("MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY", "")
    if cc_env:
        flags = []
        for cc in cc_env.split(","):
            cc = cc.strip()
            if cc:
                flags += ["-gencode", f"arch=compute_{cc},code=sm_{cc}"]
        if flags:
            return flags
    raise RuntimeError(
        "CUDA is not available and MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY not set"
    )"""


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-setup.py>", file=sys.stderr)
        sys.exit(1)

    path = pathlib.Path(sys.argv[1])
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    src = path.read_text()
    if OLD not in src:
        if "MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY" in src:
            print("[magiattn] setup.py already patched, skipping")
            return
        print(
            "Error: patch target not found in setup.py — submodule may have changed",
            file=sys.stderr,
        )
        sys.exit(1)

    path.write_text(src.replace(OLD, NEW))
    print("[magiattn] Patch applied successfully")


if __name__ == "__main__":
    main()
