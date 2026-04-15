# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance serving framework for large language models and multimodal models. It provides fast inference with RadixAttention, continuous batching, tensor/pipeline/expert/data parallelism, speculative decoding, structured outputs, and quantization (FP4/FP8/INT4/AWQ/GPTQ).

## Repository Layout

This is a monorepo with three installable packages:

- **`python/`** — Main Python package (`sglang`). Runtime engine, frontend DSL, CLI, JIT kernels.
- **`sgl-kernel/`** — Standalone C++/CUDA kernel library (`sgl_kernel`). Compiled via scikit-build-core + CMake.
- **`sgl-model-gateway/`** — Rust-based load-balancing router (`sglang-router`). Python bindings via maturin.

JIT kernels (Triton-based, compiled at runtime) live inside the main package at `python/sglang/jit_kernel/`.

## Build and Install

```bash
pip install -e "python/"                      # main package (extras: [all], [test], [diffusion], [ray])
pip install -e . --no-build-isolation         # sgl-kernel (run from sgl-kernel/)
make python-dev                               # sgl-model-gateway (run from sgl-model-gateway/)
```

## Linting

```bash
pre-commit run --all-files
# CI equivalent:
SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure
```

Hooks: isort (profile=black), ruff, black, codespell, clang-format (C++/CUDA), nbstripout. The ruff scope is intentionally minimal — only `F401` (unused imports) and `F821` (undefined names).

## Testing

Full details in `test/README.md`. Skill guides: `.claude/skills/write-sglang-test/`, `.claude/skills/ci-workflow-guide/`.

### Running Tests

```bash
python3 test/registered/core/test_srt_endpoint.py                                    # single file
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode  # single method
pytest test/registered/unit/ -v                                                       # unit tests (no GPU server)
python3 test/run_suite.py --hw cuda --suite stage-b-test-1-gpu-small                  # CI suite
```

### CI Registration

Every test file under `test/registered/` must call a registration function at module level:

```python
from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")
```

Critical rules:
- Arguments **must be literal values** (no variables, no expressions) — `run_suite.py` extracts them via AST parsing.
- Registration call must be a bare statement at module top level.
- File must end with `unittest.main()` or `pytest.main([__file__])`. No custom `argparse` or `sys.argv` modification.
- Backends: `register_cpu_ci`, `register_cuda_ci`, `register_amd_ci`, `register_npu_ci`.
- A pre-commit hook validates registration on every commit.

### Suite Selection

| Need | Suite |
|------|-------|
| No GPU | `stage-a-test-cpu` |
| 1 GPU, fits 32 GB (5090) | `stage-b-test-1-gpu-small` (most tests) |
| 1 GPU, needs Hopper/large memory | `stage-b-test-1-gpu-large` |
| JIT kernel correctness | `stage-b-kernel-unit-1-gpu-large` |
| Multi-GPU (2/4/8) | `stage-b-test-2-gpu-large`, `stage-c-test-*` |
| Long-running or experimental | `nightly-*` suites |

### Test Conventions

- E2E tests subclass `CustomTestCase` from `sglang.test.test_utils` (adds CI retry support).
- Launch servers with `popen_launch_server()` from `sglang.test.test_utils`. Reuse servers across methods via `setUpClass`.
- Default test model: `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` = `meta-llama/Llama-3.2-1B-Instruct`.
- Unit tests (no server, no model weights) go in `test/registered/unit/`, mirroring the source tree structure.
- Keep test files under 500 seconds; split if longer.

### CI Pipeline

Three sequential stages: **A** (preflight, ~3 min) → **B** (basic, ~30 min) → **C** (advanced, ~30 min). Kernel and multimodal-gen tests run in parallel with stage B. Stages gate on the previous stage passing.

## Architecture

### Runtime Process Model

Multi-process design with ZMQ-based IPC:

```
HTTP Server (FastAPI + uvicorn)  ← OpenAI / Anthropic / Ollama compatible APIs
       │
TokenizerManager (main process)  ← tokenizes requests, multimodal preprocessing
       │ ZMQ
  [DataParallelController]        ← optional, dispatches across DP workers
       │ ZMQ
Scheduler (subprocess per GPU group) ← RadixCache, batch scheduling, memory mgmt
       │
TpModelWorker → ModelRunner       ← GPU forward passes, CUDA graphs
       │ ZMQ
DetokenizerManager (subprocess)   ← token-to-text conversion → back to TokenizerManager
```

Entry points:
- **Python API**: `sglang.srt.entrypoints.engine.Engine` — in-process engine
- **HTTP server**: `sglang.srt.entrypoints.http_server.launch_server`
- **CLI**: `sglang serve` / `sglang generate` / `sglang version` (via `sglang.cli.main`)

### Batch Data Flow

Requests transform through three structures (see `schedule_batch.py`, `forward_batch_info.py`):

```
ScheduleBatch (CPU, Scheduler) → ModelWorkerBatch (CPU→GPU bridge, TpModelWorker) → ForwardBatch (GPU tensors, ModelRunner)
```

`ForwardMode.EXTEND` = prefill, `ForwardMode.DECODE` = autoregressive decoding.

### Model Registration

Models in `python/sglang/srt/models/` are auto-discovered. Each file exports `EntryClass` (a class or list of classes). `ModelRegistry` in `registry.py` maps HuggingFace architecture names to implementations via `pkgutil.iter_modules`.

### Key Subsystems

| Subsystem | Location (`python/sglang/srt/`) | Notes |
|-----------|--------------------------------|-------|
| Scheduler | `managers/scheduler.py` | 10 mixins, manages batching + cache + scheduling policy |
| KV Cache | `mem_cache/` | RadixCache for prefix sharing, memory pools, HiCache |
| Attention | `layers/attention/` | 20+ backends (FlashInfer, FlashAttention, Triton, CUTLASS MLA, etc.) |
| MoE | `layers/moe/` | Fused MoE, expert parallelism, token dispatchers |
| Quantization | `layers/quantization/` | 30+ methods (FP8, AWQ, GPTQ, FP4, etc.) |
| Speculative | `speculative/` | EAGLE (v1/v2), N-gram, DFlash |
| LoRA | `lora/` | Adapter management, batched multi-LoRA |
| Constrained | `constrained/` | Grammar-constrained decoding (XGrammar, Outlines, LLGuidance) |
| Disaggregation | `disaggregation/` | Prefill/decode separation across machines |
| Server Config | `server_args.py` | All CLI/config parameters |
| Env Config | `environ.py` | Environment variables via `EnvField` descriptors |

### Kernel Paths

- **AOT kernels** (`sgl-kernel/`): C++/CUDA compiled at install time. See `.claude/skills/add-sgl-kernel/`.
- **JIT kernels** (`python/sglang/jit_kernel/`): Triton-based, compiled at runtime. See `.claude/skills/add-jit-kernel/`.

## Frontend DSL

`python/sglang/lang/` provides the programming model: `gen()`, `select()`, `function()`, role helpers (`user`, `assistant`, `system`), `image()`, `video()`. The interpreter executes against either a local `Engine` or a remote `RuntimeEndpoint`. This is separate from the runtime (`srt/`).
