#!/bin/bash
# 编译 transfer_kernel 独立 wheel 的脚本
# 使用方法: ./build_transfer.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 清理旧的构建
rm -rf build_transfer dist_transfer

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
sed -i 's|wheel.packages = \["python/sgl_kernel"\]|wheel.packages = ["python/transfer_kernel"]|g' pyproject.toml
sed -i 's|name = "sglang-kernel"|name = "transfer-kernel"|g' pyproject.toml

# 编译 - 使用 BUILD_TRANSFER_ONLY 选项
echo "Building transfer-kernel wheel..."
pip wheel . --no-build-isolation -w dist_transfer \
    -Cbuild-dir=build_transfer \
    -Ccmake.define.BUILD_TRANSFER_ONLY=ON

echo ""
echo "Build complete! Wheel file is in dist_transfer/"
ls -la dist_transfer/*.whl 2>/dev/null || echo "No wheel file found"
