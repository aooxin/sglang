#!/bin/bash
# TODO(yingchun): 与 sgl-kernel/build_transfer.sh 重构复用
# 编译 pos_enc_kernel 独立 wheel 的脚本
# 使用方法: ./build_pos_enc.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 清理旧的构建
rm -rf build_pos_enc dist_pos_enc

# 备份原始 pyproject.toml
cp pyproject.toml pyproject.toml.bak

# 确保在脚本退出时恢复 pyproject.toml
function cleanup {
    if [ -f pyproject.toml.bak ]; then
        mv pyproject.toml.bak pyproject.toml
    fi
}
trap cleanup EXIT

# 临时修改 pyproject.toml 的 wheel.packages 和 project.name
sed -i 's|wheel.packages = \["python/sgl_kernel"\]|wheel.packages = ["python/pos_enc_kernel"]|g' pyproject.toml
sed -i 's|name = "sglang-kernel"|name = "pos_enc_kernel"|g' pyproject.toml

# 编译 - 使用 BUILD_POS_ENC_ONLY 选项
echo "Building pos-enc-kernel wheel..."
pip wheel . --no-build-isolation -w dist_pos_enc \
    -Cbuild-dir=build_pos_enc \
    -Ccmake.define.BUILD_POS_ENC_ONLY=ON

echo ""
echo "Build complete! Wheel file is in dist_pos_enc/"
ls -la dist_pos_enc/*.whl 2>/dev/null || echo "No wheel file found"
