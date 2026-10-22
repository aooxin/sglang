#!/usr/bin/env bash
# 离线编译主模型及 DFLASH 的 DeepGEMM kernel cache，并打包供部署使用。
#
# Usage:
#   bash scripts/compile_deepgemm_cache.sh CACHE_DIR OUTPUT.tar.gz -- [SGLang 参数]
#   bash scripts/compile_deepgemm_cache.sh --help
#
# Example: MiMo v2.6 Pro，单机 403 TP8/EP8、48K prefill + FP8 DFLASH。
# 模型路径、量化、并行、MoE/attention 后端及投机参数须与实际服务保持一致。
#   bash scripts/compile_deepgemm_cache.sh /mnt/deepgemm-build /tmp/mimo-pro-cache.tar.gz -- \
#     --model-path /models/mimo-v2.6-pro \
#     --speculative-draft-model-path /models/mimo-v2.6-pro-dflash \
#     --tp-size 8 --ep-size 8 --moe-dense-tp-size 1 \
#     --moe-a2a-backend deepep --deepep-mode normal \
#     --disaggregation-mode prefill --chunked-prefill-size 49152 \
#     --quantization fp8 --speculative-draft-model-quantization fp8 \
#     --speculative-algorithm DFLASH --speculative-num-draft-tokens 8 \
#     --attention-backend fa3 --speculative-draft-attention-backend fa3 \
#     --page-size 64 --context-length 1048576 --trust-remote-code \
#     --mem-fraction-static 0.65 --host 127.0.0.1 --port 39001 --timeout 14400
#
# Example: 在同一 cache 目录继续补齐 Flash，生成 Pro + Flash 合集。
#   SGLANG_MIMO_V2_BF16_MOE_GATE=1 \
#   bash scripts/compile_deepgemm_cache.sh /mnt/deepgemm-build /tmp/mimo-pro-flash-cache.tar.gz -- \
#     --model-path /models/mimo-v2.6-flash \
#     --speculative-draft-model-path /models/mimo-v2.6-flash-dflash \
#     --tp-size 8 --dp-size 2 --ep-size 8 --enable-dp-attention \
#     --moe-dense-tp-size 1 --enable-dp-lm-head --moe-a2a-backend megamoe \
#     --disaggregation-mode prefill --chunked-prefill-size 32768 \
#     --quantization fp8 --speculative-draft-model-quantization fp8 \
#     --speculative-algorithm DFLASH --speculative-num-draft-tokens 8 \
#     --attention-backend fa3 --speculative-draft-attention-backend fa3 \
#     --page-size 64 --context-length 1048576 --trust-remote-code \
#     --mem-fraction-static 0.65 --host 127.0.0.1 --port 39001 --timeout 14400
#
# 说明：
# - 在已安装 SGLang/DeepGEMM 的单机离线环境执行；GPU、DeepGEMM 构建和 SM 设置
#   须与服务一致。使用独立且可写的 CACHE_DIR，不要与运行中的服务共用。
# - 参数原样传给 sglang.compile_deep_gemm；忽略 SGLANG_LOGGING_CONFIG_PATH，
#   日志级别固定为 info，以保证编译范围检查能收到完整日志。
#   小请求负责发现 kernel 类型及 N/K，随后遍历该启动配置对应的 M 范围。
# - 离线编译可降低 --mem-fraction-static 为临时张量留空间；若检测到范围缩减，
#   本脚本拒绝打包，调整显存比例后可在同一目录重试。不同模型/配置需分别执行。
# - 包仅覆盖实际经过的 kernel 路径及配置范围，上线仍需冷启动请求验证。
# - 包内根目录为 cache/ 和 compile-info/；部署时解压到目标目录，然后将
#   SGLANG_DG_CACHE_DIR 指向该目录（不是其中的 cache/ 子目录），无需 strip-components。
# - 包内仅附本次编译的环境摘要；完整命令和日志保留在本地，不随包分发。

set -euo pipefail

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ $# -lt 4 || ${3:-} != -- ]]; then
    usage >&2
    exit 2
fi

cache_dir=$(realpath -m -- "$1")
archive=$(realpath -m -- "$2")
shift 3
[[ $archive == *.tar.gz ]] || die '输出文件必须以 .tar.gz 结尾。'
case "$archive" in
    "$cache_dir/cache/"*|"$cache_dir/compile-info/"*)
        die '输出文件不能放在待打包的 cache/ 或 compile-info/ 内。'
        ;;
esac

# 创建目录和锁文件前检查 GPU，避免检查失败时留下无效的运行目录。
python3 - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("离线编译需要可用的 CUDA GPU。")
PY

mkdir -p -- "$cache_dir/compile-info" "$(dirname -- "$archive")"
# 不同 cache 目录也可能写同一个输出包；锁文件需保留，避免并发时锁住不同 inode。
exec 8>"$archive.lock"
flock -n 8 || die "该输出路径已有编译任务：$archive"
[[ ! -e $archive && ! -e $archive.sha256 && ! -L $archive.sha256 ]] \
    || die "输出已存在，请使用新的文件名：$archive"
exec 9>"$cache_dir/.compile.lock"
flock -n 9 || die "该 cache 目录已有编译任务：$cache_dir"
run_dir=$(mktemp -d "$cache_dir/compile-info/run-XXXXXXXX")

# 必须在导入 SGLang/DeepGEMM 前设置，避免继承服务的禁用/快速预热选项。
export SGLANG_DG_CACHE_DIR="$cache_dir"
export DG_JIT_CACHE_DIR="$cache_dir"
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=1
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=0
unset SGLANG_LOGGING_CONFIG_PATH

command_args=(python3 -u -m sglang.compile_deep_gemm "$@" --log-level info)
{
    printf '%q ' "${command_args[@]}"
    printf '\n'
} > "$run_dir/command.txt"
{
    date -u '+UTC: %Y-%m-%dT%H:%M:%SZ'
    printf 'SGLANG_DG_CACHE_DIR=%s\n' "$cache_dir"
    printf 'SGLANG_ENABLE_JIT_DEEPGEMM=1\nSGLANG_JIT_DEEPGEMM_PRECOMPILE=1\nSGLANG_JIT_DEEPGEMM_FAST_WARMUP=0\n'
    python3 - <<'PY'
import importlib.metadata
import importlib.util
import os

import torch

print("sglang_path:", importlib.util.find_spec("sglang").origin)
for name in ("sglang", "deep-gemm", "torch"):
    print(f"{name}: {importlib.metadata.version(name)}")
print("torch_cuda:", torch.version.cuda)
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"gpu[{index}]: {props.name}, sm_{props.major}{props.minor}, SMs={props.multi_processor_count}")
for key in ("CUDA_VISIBLE_DEVICES", "SGL_DG_USE_NVRTC", "DG_JIT_NVCC_COMPILER",
            "SGLANG_MIMO_V2_BF16_MOE_GATE"):
    print(f"{key}: {os.getenv(key, '<unset>')}")
PY
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version
    fi
} > "$run_dir/environment.txt"

printf '编译日志：%s\n' "$run_dir/compile.log"
if ! "${command_args[@]}" 2>&1 | tee "$run_dir/compile.log"; then
    die "编译失败，未打包；请查看 $run_dir/compile.log"
fi

# 当前 Python 编译入口不生成状态文件，需结合退出码和本次日志检查。
# 以下文案来自 compile_deep_gemm.py 和 deep_gemm_wrapper/compile_utils.py；
# 升级时需同步核对，尤其是显存不足时缩减 M 范围的 warning（入口仍可能正常退出）。
if grep -Fq 'reducing max_m to ' "$run_dir/compile.log"; then
    die '编译范围因显存不足而缩减，未打包；请降低 --mem-fraction-static 后重试。'
fi
grep -Fq 'DeepGEMM Kernels compilation finished successfully.' "$run_dir/compile.log" \
    || die "未确认编译完成，未打包；请查看 $run_dir/compile.log"
grep -Fq 'Try DeepGEMM JIT Compiling for ' "$run_dir/compile.log" \
    || die '本次未发现 DeepGEMM 预编译路径，未打包；请检查模型、量化及后端配置。'
[[ -d $cache_dir/cache ]] || die "未生成 cache 子目录：$cache_dir"
kernel_count=$(find "$cache_dir/cache" -type f -name kernel.cubin -printf '.' | wc -c)
[[ $kernel_count -gt 0 ]] || die '没有找到 kernel.cubin，未打包。'
printf 'kernel_count=%s\n' "$kernel_count" >> "$run_dir/environment.txt"

# 完整生成后再使用最终文件名，避免把中断的压缩包当成可部署产物。
temporary_archive=$(mktemp "${archive}.partial.XXXXXXXX")
trap 'rm -f -- "$temporary_archive"' EXIT
# 仅打包明确筛选的环境摘要；当前及历史运行的命令、日志可能包含凭据。
tar -czf "$temporary_archive" -C "$cache_dir" cache "${run_dir#"$cache_dir/"}/environment.txt"
# 同目录硬链接原子发布，若编译期间出现同名文件则失败，不覆盖现有包。
ln -T -- "$temporary_archive" "$archive" || die "输出包已存在或无法发布：$archive"
(
    set -o noclobber
    cd -- "$(dirname -- "$archive")"
    sha256sum -- "$(basename -- "$archive")" > "$(basename -- "$archive").sha256"
)
printf '已打包 %s 个 kernel：%s\nSHA256：%s.sha256\n' "$kernel_count" "$archive" "$archive"
