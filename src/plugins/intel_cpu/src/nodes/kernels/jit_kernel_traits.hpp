// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <dnnl_types.h>
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
namespace x64 = dnnl::impl::cpu::x64;

template<x64::cpu_isa_t isa, ov::element::Type_t element_type>
class jit_kernel_traits {
public:
    using Vmm = typename conditional3<isa == x64::sse41,
                                      Xbyak::Xmm,
                                      isa == x64::avx2,
                                      Xbyak::Ymm,
                                      Xbyak::Zmm>::type;

    static constexpr unsigned VCMPPS_LE = 0x02;
    static constexpr unsigned VCMPPS_GT = 0x0e;
    static constexpr unsigned SIMD_WIDTH = x64::cpu_isa_traits<isa>::vlen / sizeof(typename ov::element_type_traits<element_type>::value_type);
};

} // namespace intel_cpu
} // namespace ov
