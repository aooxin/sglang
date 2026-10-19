/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "sgl_kernel_torch_shim.h"

// Keep the 0520 operator schema while compiling the unmodified 0510 backend.
// Only the two new arguments are handled here; existing conversions (including
// optional<const Tensor>& for sinks) remain in make_pytorch_shim.
template <typename Ret, typename... Args>
auto make_fa3_0510_pytorch_shim(Ret (*fun)(Args...)) {
  return [shim = make_pytorch_shim(fun)](
             pytorch_library_compatible_type_t<Args>... args,
             const std::optional<at::Tensor>& sparse_mask_fine,
             bool only_qv) {
    TORCH_CHECK(!sparse_mask_fine.has_value(), "0510 FA3 does not support sparse_mask_fine");
    TORCH_CHECK(!only_qv, "0510 FA3 does not support only_qv=True");
    return shim(args...);
  };
}
