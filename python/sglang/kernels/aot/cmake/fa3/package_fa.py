#!/usr/bin/env python3
"""Validate a local FA3 build and package only its extension, without installing."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path


def describe_torch():
    import torch
    from wheel.wheelfile import WheelFile  # noqa: F401 -- fail before compilation

    if (
        sys.version_info < (3, 10)
        or sys.platform != "linux"
        or platform.python_implementation() != "CPython"
    ):
        raise RuntimeError("FA3 requires CPython >=3.10 on Linux")
    if not str(torch.__version__).startswith("2.13.") or not torch.version.cuda:
        raise RuntimeError(
            f"0520 FA3 requires CUDA-enabled Torch 2.13; got {torch.__version__}"
        )
    return {
        "torch_root": str(Path(torch.__file__).resolve().parent),
        "torch_version": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cxx11_abi": int(torch.compiled_with_cxx11_abi()),
        "python": sys.executable,
    }


def package_fa(library, build_info, dist_dir):
    import torch
    from wheel.wheelfile import WheelFile

    info = json.loads(build_info.read_text())
    for key, value in describe_torch().items():
        if info.get(key) != value:
            raise RuntimeError(
                f"FA3 build environment changed: {key}: built={info.get(key)!r}, current={value!r}"
            )
    library = library.resolve(strict=True)
    # Run in this fresh process, not by importing sgl_kernel, which may select
    # other installed kernels or query a GPU during package initialization.
    print(f"[fa3-build] checking native loading: {library}", flush=True)
    torch.ops.load_library(str(library))
    schema = torch.ops.sgl_kernel.fwd.default._schema
    names = tuple(arg.name for arg in schema.arguments)
    if len(names) != 37 or names[-3:] != ("sinks", "sparse_mask_fine", "only_qv"):
        raise RuntimeError(f"Unexpected 0520 FA3 fwd schema: {schema}")
    torch.ops.sgl_kernel.get_scheduler_metadata.default._schema
    print(
        "[fa3-build] native load/schema OK; hashing and packaging the extension",
        flush=True,
    )

    sha256 = hashlib.sha256()
    with library.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
    manifest = {
        **info,
        "sha256": sha256.hexdigest(),
        "bytes": library.stat().st_size,
        "fwd_schema": str(schema),
        "gpu_validation": "not_run",
    }

    # A distinct distribution name avoids representing this single extension
    # as a complete sglang-kernel wheel. CUDA/Torch details travel with it.
    distribution = "sglang_kernel_fa3_cu129"
    version = "0.0.0+torch" + re.sub(r"[^a-z0-9.]+", ".", info["torch_version"].lower())
    version += ".nvcc" + info["nvcc"]
    tag = "cp310-abi3-" + sysconfig.get_platform().replace("-", "_").replace(".", "_")
    wheel_name = f"{distribution}-{version}-{tag}.whl"
    metadata_dir = f"{distribution}-{version}.dist-info"
    dist_dir.mkdir(parents=True, exist_ok=True)
    # Finish validation and packaging before publishing any outputs. A failed
    # native check leaves the previous successful artifacts untouched.
    with tempfile.TemporaryDirectory(prefix=".fa3-package-", dir=dist_dir) as temp:
        temp = Path(temp)
        with WheelFile(temp / wheel_name, "w") as wheel:
            wheel.write(library, "sgl_kernel/flash_ops.abi3.so")
            wheel.writestr(
                f"{metadata_dir}/METADATA",
                "Metadata-Version: 2.1\n"
                "Name: sglang-kernel-fa3-cu129\n"
                f"Version: {version}\n"
                "Summary: Local FA3-only extension for an existing sgl_kernel installation\n"
                "Requires-Python: >=3.10\n"
                f"Requires-Dist: torch=={info['torch_version']}\n\n",
            )
            wheel.writestr(
                f"{metadata_dir}/WHEEL",
                f"Wheel-Version: 1.0\nGenerator: build_fa.sh\nRoot-Is-Purelib: false\nTag: {tag}\n",
            )
            wheel.writestr(
                f"{metadata_dir}/fa3-build.json", json.dumps(manifest, indent=2) + "\n"
            )
        shutil.copy2(library, temp / "flash_ops.abi3.so")
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for artifact in temp.iterdir():
            os.replace(artifact, dist_dir / artifact.name)
    print(f"[fa3-build] SHA256={manifest['sha256']}", flush=True)
    print(f"[fa3-build] wheel: {dist_dir / wheel_name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--build-info", type=Path)
    parser.add_argument("--dist-dir", type=Path)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe_torch()))
    elif args.library and args.build_info and args.dist_dir:
        package_fa(args.library, args.build_info, args.dist_dir)
    else:
        parser.error("use --describe or provide --library, --build-info and --dist-dir")


if __name__ == "__main__":
    main()
