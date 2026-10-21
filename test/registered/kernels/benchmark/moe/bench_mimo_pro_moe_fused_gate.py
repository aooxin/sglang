import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.moe.mimo_pro_moe_fused_gate import (
    MIMO_PRO_NUM_EXPERTS,
    mimo_pro_moe_fused_gate,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

TOPK = 8


# 512 is the last token count of the small-batch kernel. Eager timing keeps the
# per-call launch cost, which dominates eager prefill over ~70 MoE layers.
@marker.parametrize("use_cuda_graph", [True, False])
@marker.parametrize("num_tokens", [1, 16, 128, 512, 513, 2048, 8192], [16, 2048])
@marker.benchmark("provider", ["mimo_pro", "triton"])
def benchmark(num_tokens: int, use_cuda_graph: bool, provider: str):
    torch.manual_seed(0)
    logits = create_random(num_tokens, MIMO_PRO_NUM_EXPERTS, dtype=torch.float32)
    bias = create_random(MIMO_PRO_NUM_EXPERTS, dtype=torch.float32)
    if provider == "mimo_pro":
        return marker.do_bench(
            mimo_pro_moe_fused_gate,
            input_args=(logits, bias, TOPK),
            input_kwargs=dict(renormalize=True),
            use_cuda_graph=use_cuda_graph,
        )
    if provider == "triton":
        return marker.do_bench(
            moe_fused_gate,
            input_args=(logits, bias, TOPK),
            input_kwargs=dict(scoring_func="sigmoid", renormalize=True),
            use_cuda_graph=use_cuda_graph,
        )
    raise ValueError(f"unknown provider: {provider}")


if __name__ == "__main__":
    benchmark.run()
