"""
性能对比测试：CUDA Kernel vs PyTorch 实现（256/1 配置）

测试 moe_fused_gate CUDA kernel 和 PyTorch 原生实现在 256/1 配置下的性能差异。

配置说明：
- 256 experts / 1 group = VPT 256
- 这是最高行并行度配置
- 测试不同批处理大小的性能
"""

import argparse
import json
import time
from typing import Tuple

import numpy as np
import torch

# 尝试导入 CUDA 扩展
try:
    import moe_cuda
    HAS_CUDA_KERNEL = True
    print(f"✅ Successfully loaded moe_cuda extension (version {moe_cuda.__version__})")
except ImportError:
    HAS_CUDA_KERNEL = False
    print("⚠️  moe_cuda extension not found. Will only test PyTorch implementation.")
    print("   To test CUDA kernel, please compile: pip install -e .")


def torch_biased_grouped_topk_impl(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch 实现的 biased grouped topk（参考 SGLang 代码）

    注意：为了数值稳定性，在 float32 精度下计算
    """
    # 转换到 float32 以获得更好的数值稳定性
    gating_output_fp32 = gating_output.to(torch.float32)
    correction_bias_fp32 = correction_bias.to(torch.float32)

    scores = gating_output_fp32.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    # Add bias
    scores_for_choice = scores.view(num_token, -1) + correction_bias_fp32.unsqueeze(0)

    # Group selection
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )

    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

    # Top-k selection
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    # Handle fused shared experts
    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    # Renormalize
    if renormalize:
        if num_fused_shared_experts == 0:
            topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights_sum = topk_weights[:, :-1].sum(dim=-1, keepdim=True)

        # 避免除以零
        topk_weights_sum = torch.clamp(topk_weights_sum, min=1e-9)
        topk_weights = topk_weights / topk_weights_sum

        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    # 确保返回 float32
    return topk_weights, topk_ids.to(torch.int32)


def cuda_kernel_biased_grouped_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,  # 接受但忽略（kernel 内部处理）
    num_expert_group: int,
    topk_group: int,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA kernel 实现的 biased grouped topk

    注意：
    1. renormalize 参数被接受但忽略，因为 CUDA kernel 内部已处理归一化
    2. CUDA kernel 要求 float32 输入，所以在这里转换
    """
    # 转换到 float32（CUDA kernel 要求）
    gating_output_fp32 = gating_output.to(torch.float32)
    correction_bias_fp32 = correction_bias.to(torch.float32)

    topk_weights, topk_ids = moe_cuda.moe_fused_gate(
        gating_output_fp32,
        correction_bias_fp32,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
    return topk_weights, topk_ids


def benchmark_implementation(
    impl_name: str,
    impl_func,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    config: dict,
    warmup_iters: int = 20,
    test_iters: int = 100,
) -> dict:
    """
    测试单个实现的性能
    """
    print(f"\n{'='*60}")
    print(f"Testing: {impl_name}")
    print(f"{'='*60}")

    # 预热
    for _ in range(warmup_iters):
        topk_weights, topk_ids = impl_func(
            gating_output,
            correction_bias,
            **config
        )

    torch.cuda.synchronize()

    # 测试
    times = []
    for _ in range(test_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        topk_weights, topk_ids = impl_func(
            gating_output,
            correction_bias,
            **config
        )

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # ms

    # 统计
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p50_time = np.percentile(times, 50)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)

    batch_size = gating_output.shape[0]
    throughput = batch_size / (avg_time / 1000)  # rows/sec

    result = {
        'implementation': impl_name,
        'batch_size': batch_size,
        'avg_time_ms': float(avg_time),
        'std_time_ms': float(std_time),
        'min_time_ms': float(min_time),
        'max_time_ms': float(max_time),
        'p50_time_ms': float(p50_time),
        'p95_time_ms': float(p95_time),
        'p99_time_ms': float(p99_time),
        'throughput_rows_per_sec': float(throughput),
    }

    # 打印结果
    print(f"  Avg Time: {avg_time:.3f} ms (±{std_time:.3f})")
    print(f"  Min/Max:  {min_time:.3f} / {max_time:.3f} ms")
    print(f"  P50/P95/P99: {p50_time:.3f} / {p95_time:.3f} / {p99_time:.3f} ms")
    print(f"  Throughput: {throughput:,.0f} rows/sec")

    return result


def verify_correctness(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    config: dict,
    torch_result: Tuple[torch.Tensor, torch.Tensor],
    cuda_result: Tuple[torch.Tensor, torch.Tensor],
    atol: float = 1e-3,
) -> bool:
    """
    验证两种实现的结果是否一致
    """
    torch_weights, torch_ids = torch_result
    cuda_weights, cuda_ids = cuda_result

    print(f"\n{'='*60}")
    print("Correctness Verification")
    print(f"{'='*60}")

    # 检查形状
    assert torch_weights.shape == cuda_weights.shape, \
        f"Shape mismatch: {torch_weights.shape} vs {cuda_weights.shape}"
    assert torch_ids.shape == cuda_ids.shape, \
        f"Shape mismatch: {torch_ids.shape} vs {cuda_ids.shape}"
    print(f"✅ Shape match: {torch_weights.shape}")

    # 检查索引（排序后比较）
    # 注意：由于浮点精度和并行计算顺序，索引可能以不同顺序出现
    # 排序后再比较，因为顺序不重要，只要选择了相同的专家即可
    torch_ids_sorted, torch_sort_indices = torch.sort(torch_ids, dim=1)
    cuda_ids_sorted, cuda_sort_indices = torch.sort(cuda_ids, dim=1)
    ids_match = (torch_ids_sorted == cuda_ids_sorted).float().mean().item()
    print(f"  Indices match ratio (sorted): {ids_match*100:.2f}%")

    # 检查权重（按照排序后的索引顺序重排）
    # 使用 gather 按照排序索引重新排列权重
    torch_weights_sorted = torch.gather(torch_weights, 1, torch_sort_indices)
    cuda_weights_sorted = torch.gather(cuda_weights, 1, cuda_sort_indices)

    weights_close = torch.allclose(torch_weights_sorted, cuda_weights_sorted, atol=atol)
    max_weight_diff = (torch_weights_sorted - cuda_weights_sorted).abs().max().item()
    mean_weight_diff = (torch_weights_sorted - cuda_weights_sorted).abs().mean().item()

    print(f"  Weights close (atol={atol}, sorted): {weights_close}")
    print(f"  Max weight diff: {max_weight_diff:.6f}")
    print(f"  Mean weight diff: {mean_weight_diff:.6f}")

    # 检查权重和是否为1
    torch_sum = torch_weights.sum(dim=1)
    cuda_sum = cuda_weights.sum(dim=1)
    torch_sum_close = torch.allclose(torch_sum, torch.ones_like(torch_sum), atol=atol)
    cuda_sum_close = torch.allclose(cuda_sum, torch.ones_like(cuda_sum), atol=atol)

    print(f"  PyTorch weights sum to 1: {torch_sum_close}")
    print(f"  CUDA weights sum to 1: {cuda_sum_close}")

    # 总体判断
    is_correct = weights_close and ids_match > 0.90  # 90% 索引匹配

    if is_correct:
        print(f"\n✅ Correctness check PASSED")
    else:
        print(f"\n⚠️  Correctness check FAILED")
        print(f"   Weights close: {weights_close}")
        print(f"   Indices match: {ids_match*100:.2f}% (expected >90%)")

    return is_correct


def run_comparison(
    batch_sizes: list,
    num_experts: int = 256,
    num_expert_group: int = 1,
    topk: int = 8,
    topk_group: int = 1,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    renormalize: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    warmup_iters: int = 20,
    test_iters: int = 100,
    output_file: str = None,
):
    """
    运行完整的对比测试
    """
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return

    vpt = num_experts // num_expert_group

    print("="*80)
    print("MoE Gate Performance Comparison: CUDA Kernel vs PyTorch")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Num Experts: {num_experts}")
    print(f"  Num Expert Groups: {num_expert_group}")
    print(f"  VPT: {vpt}")
    print(f"  TopK: {topk}")
    print(f"  TopK Group: {topk_group}")
    print(f"  Data Type: {dtype}")
    print(f"  Renormalize: {renormalize}")
    print(f"  Routed Scaling Factor: {routed_scaling_factor}")
    print(f"\nTest Parameters:")
    print(f"  Warmup Iterations: {warmup_iters}")
    print(f"  Test Iterations: {test_iters}")
    print(f"  Batch Sizes: {batch_sizes}")
    print()

    config = {
        'topk': topk,
        'renormalize': renormalize,
        'num_expert_group': num_expert_group,
        'topk_group': topk_group,
        'num_fused_shared_experts': num_fused_shared_experts,
        'routed_scaling_factor': routed_scaling_factor,
        'apply_routed_scaling_factor_on_output': apply_routed_scaling_factor_on_output,
    }

    all_results = []

    for batch_size in batch_sizes:
        print(f"\n{'#'*80}")
        print(f"# Batch Size: {batch_size}")
        print(f"{'#'*80}")

        # 创建输入数据
        gating_output = torch.randn(
            batch_size, num_experts,
            dtype=dtype,
            device='cuda'
        )
        correction_bias = torch.randn(
            num_experts,
            dtype=dtype,
            device='cuda'
        )

        batch_results = {
            'batch_size': batch_size,
            'config': {
                'num_experts': num_experts,
                'num_expert_group': num_expert_group,
                'vpt': vpt,
                'topk': topk,
                'topk_group': topk_group,
                'dtype': str(dtype),
            }
        }

        # 测试 PyTorch 实现
        torch_result_tuple = None
        torch_time_result = benchmark_implementation(
            "PyTorch Implementation",
            torch_biased_grouped_topk_impl,
            gating_output,
            correction_bias,
            config,
            warmup_iters,
            test_iters,
        )
        batch_results['pytorch'] = torch_time_result

        # 保存一个结果用于正确性验证
        torch_result_tuple = torch_biased_grouped_topk_impl(
            gating_output, correction_bias, **config
        )

        # 测试 CUDA kernel（如果可用）
        if HAS_CUDA_KERNEL:
            try:
                cuda_result_tuple = None
                cuda_time_result = benchmark_implementation(
                    "CUDA Kernel",
                    cuda_kernel_biased_grouped_topk,
                    gating_output,
                    correction_bias,
                    config,
                    warmup_iters,
                    test_iters,
                )
                batch_results['cuda'] = cuda_time_result

                # 保存结果用于正确性验证
                cuda_result_tuple = cuda_kernel_biased_grouped_topk(
                    gating_output, correction_bias, **config
                )

                # 验证正确性
                if torch_result_tuple and cuda_result_tuple:
                    is_correct = verify_correctness(
                        gating_output,
                        correction_bias,
                        config,
                        torch_result_tuple,
                        cuda_result_tuple,
                    )
                    batch_results['correctness'] = is_correct

                # 计算加速比
                speedup = torch_time_result['avg_time_ms'] / cuda_time_result['avg_time_ms']
                speedup_pct = (speedup - 1.0) * 100
                batch_results['speedup'] = {
                    'ratio': float(speedup),
                    'percentage': float(speedup_pct),
                }

                print(f"\n{'='*60}")
                print("Performance Comparison")
                print(f"{'='*60}")
                print(f"  PyTorch: {torch_time_result['avg_time_ms']:.3f} ms")
                print(f"  CUDA:    {cuda_time_result['avg_time_ms']:.3f} ms")
                print(f"  Speedup: {speedup:.2f}x ({speedup_pct:+.1f}%)")

                if speedup > 1.0:
                    print(f"  ✅ CUDA is {speedup:.2f}x faster")
                else:
                    print(f"  ⚠️  PyTorch is {1/speedup:.2f}x faster")

            except Exception as e:
                print(f"❌ CUDA kernel test failed: {e}")
                import traceback
                traceback.print_exc()
                batch_results['cuda'] = {'error': str(e)}

        all_results.append(batch_results)

    # 打印总结
    print_summary(all_results)

    # 保存结果
    if output_file:
        save_results(all_results, output_file)

    return all_results


def print_summary(results: list):
    """打印测试总结"""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # 表头
    print(f"\n{'Batch':<8} {'PyTorch (ms)':<15} {'CUDA (ms)':<15} {'Speedup':<12} {'Correct':<10}")
    print('-'*65)

    for result in results:
        batch_size = result['batch_size']
        pytorch_time = result['pytorch']['avg_time_ms']

        if 'cuda' in result and 'avg_time_ms' in result['cuda']:
            cuda_time = result['cuda']['avg_time_ms']
            speedup = result['speedup']['ratio']
            speedup_str = f"{speedup:.2f}x"

            # 标记
            if speedup > 1.2:
                marker = "✅"
            elif speedup > 1.0:
                marker = "✓"
            else:
                marker = "⚠️"

            correct = result.get('correctness', False)
            correct_str = "✅" if correct else "❌"

            print(f"{batch_size:<8} {pytorch_time:>10.3f}      {cuda_time:>10.3f}      "
                  f"{marker} {speedup_str:<9} {correct_str}")
        else:
            print(f"{batch_size:<8} {pytorch_time:>10.3f}      {'N/A':<15} {'N/A':<12} {'N/A'}")

    # 平均加速比
    if any('speedup' in r for r in results):
        speedups = [r['speedup']['ratio'] for r in results if 'speedup' in r]
        avg_speedup = np.mean(speedups)
        print(f"\n{'='*65}")
        print(f"Average Speedup: {avg_speedup:.2f}x")


def save_results(results: list, filename: str):
    """保存结果到 JSON 文件"""
    output = {
        'gpu': torch.cuda.get_device_name(0),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'has_cuda_kernel': HAS_CUDA_KERNEL,
        'results': results,
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare CUDA kernel vs PyTorch for MoE Gate (256/1 config)'
    )
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[128, 512, 2048, 8192],
                       help='Batch sizes to test')
    parser.add_argument('--num-experts', type=int, default=256,
                       help='Number of experts')
    parser.add_argument('--num-expert-group', type=int, default=1,
                       help='Number of expert groups (256/1 means VPT=256)')
    parser.add_argument('--topk', type=int, default=8,
                       help='Number of experts to select')
    parser.add_argument('--topk-group', type=int, default=1,
                       help='Number of groups to select')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Data type')
    parser.add_argument('--warmup-iters', type=int, default=20,
                       help='Warmup iterations')
    parser.add_argument('--test-iters', type=int, default=100,
                       help='Test iterations')
    parser.add_argument('--output', type=str, default='cuda_vs_torch_results.json',
                       help='Output JSON file')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer iterations')

    args = parser.parse_args()

    # 转换数据类型
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # 快速测试模式
    if args.quick:
        batch_sizes = [512, 2048]
        warmup_iters = 5
        test_iters = 20
        print("⚡ Quick test mode enabled")
    else:
        batch_sizes = args.batch_sizes
        warmup_iters = args.warmup_iters
        test_iters = args.test_iters

    # 运行对比测试
    run_comparison(
        batch_sizes=batch_sizes,
        num_experts=args.num_experts,
        num_expert_group=args.num_expert_group,
        topk=args.topk,
        topk_group=args.topk_group,
        dtype=dtype,
        warmup_iters=warmup_iters,
        test_iters=test_iters,
        output_file=args.output,
    )


if __name__ == '__main__':
    main()
