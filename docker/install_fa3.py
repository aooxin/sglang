"""Install the mounted 0510-kernel FA3 build before launching an SGLang server."""

import fcntl
import hashlib
import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Keep in sync with ENV FA3_ARTIFACT_DIR in docker/Dockerfile.xiaomi, which wins.
DEFAULT_ARTIFACT_DIR = "/mnt/xiaoxin7-dev/fa3"
DEFAULT_SHA256 = "bb991e61860786478570655e13e2c6e356cfa049c900025608a9f04e0742983c"

NATIVE_CHECK = r"""
import sys
import torch

version = str(torch.__version__)
cuda = torch.version.cuda
abi = torch.compiled_with_cxx11_abi()
print(f"[fa3-install] torch={version}, CUDA={cuda}, CXX11 ABI={abi}", flush=True)
if version != "2.13.0+cu130" or cuda != "13.0" or not abi:
    raise RuntimeError("FA3 requires Torch 2.13.0+cu130, CUDA 13.0 and CXX11 ABI=True")
torch.ops.load_library(sys.argv[1])
schema = torch.ops.sgl_kernel.fwd.default._schema
names = tuple(arg.name for arg in schema.arguments)
if len(names) != 37 or names[-3:] != ("sinks", "sparse_mask_fine", "only_qv"):
    raise RuntimeError(f"Unexpected 0520 FA3 fwd schema: {schema}")
torch.ops.sgl_kernel.get_scheduler_metadata.default._schema
print("[fa3-install] native load/schema OK (37 fwd arguments)", flush=True)
"""


def installed_library():
    # Importing sgl_kernel initializes other kernels/GPU state. Discover its
    # active extension without importing the package or loading the old FA3.
    package = importlib.util.find_spec("sgl_kernel")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError(f"sgl_kernel is not installed for {sys.executable}")
    extension = importlib.machinery.PathFinder.find_spec(
        "sgl_kernel.flash_ops", package.submodule_search_locations
    )
    if extension is None or not isinstance(
        extension.loader, importlib.machinery.ExtensionFileLoader
    ):
        raise RuntimeError("Cannot locate the installed sgl_kernel.flash_ops extension")
    # Preserve Python's suffix precedence: a cpython-312 build can shadow abi3.
    return Path(extension.origin)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_library(path):
    # A fresh process ensures the old extension cannot supply missing symbols
    # or operator registrations. This checks loading; it does not execute GPUs.
    subprocess.run([sys.executable, "-c", NATIVE_CHECK, str(path)], check=True)


def install(source, expected_sha256):
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha256
    ):
        raise RuntimeError("FA3_ARTIFACT_SHA256 must contain a SHA256 hex digest")
    if not source.is_file():
        raise RuntimeError(
            f"FA3 artifact missing: {source}; check the cloud disk mount"
        )
    library = installed_library()
    print(f"[fa3-install] source: {source}", flush=True)
    print(f"[fa3-install] active extension: {library}", flush=True)

    # Multiple startup invocations in one container must preserve the same
    # original backup. Different service pods each install their own local copy.
    with (library.parent / ".fa3-install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if digest(library) == expected_sha256:
            validate_library(library)
            print("[fa3-install] verified build already installed", flush=True)
        else:
            # Stage on the destination filesystem for an atomic rename. Never
            # truncate the existing .so, which another process may have mapped.
            fd, name = tempfile.mkstemp(
                prefix=".fa3-install-", suffix=".so", dir=library.parent
            )
            os.close(fd)
            staged = Path(name)
            try:
                shutil.copyfile(source, staged)
                actual_sha256 = digest(staged)
                if actual_sha256 != expected_sha256:
                    raise RuntimeError(
                        f"FA3 SHA256 mismatch: expected {expected_sha256}, "
                        f"got {actual_sha256}; installed library was not changed"
                    )
                shutil.copymode(library, staged)
                validate_library(staged)
                backup = library.with_name(library.name + ".before_fa3_cu129")
                if not backup.exists():
                    try:
                        os.link(library, backup)
                    except OSError:
                        shutil.copy2(library, backup)
                os.replace(staged, library)
                print(f"[fa3-install] original backup: {backup}", flush=True)
            finally:
                staged.unlink(missing_ok=True)
    print(f"[fa3-install] ready: {library}; SHA256={expected_sha256}", flush=True)


def main():
    directory = Path(os.environ.get("FA3_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))
    expected_sha256 = os.environ.get("FA3_ARTIFACT_SHA256", DEFAULT_SHA256).lower()
    try:
        install(directory.expanduser().resolve() / "flash_ops.abi3.so", expected_sha256)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[fa3-install] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
