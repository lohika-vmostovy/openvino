// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <math.h>
#include <dnnl_extension_utils.h>
#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include "jit_kernel_base.hpp"

namespace ov {
namespace intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;

struct jit_kernel_args_stub {};

template<typename jit_params, typename jit_call_args>
class jit_kernel : public jit_kernel_base {
public:
    using kernel_func = void (*)(const jit_call_args *);

    explicit jit_kernel(x64::cpu_isa_t max_cpu_isa, const jit_params &jqp)
            : jit_kernel_base{max_cpu_isa}, kernel_{nullptr}, params_{jqp} {}
    ~jit_kernel() override = default;

    status_t create_kernel() override {
        const status_t code = jit_generator::create_kernel();
        if (code != dnnl::impl::status::success) {
            IE_THROW() << "Could not create kernel. Error code: " << std::to_string(code) << ". " <<
                       "Xbyak error code: " << Xbyak::ConvertErrorToString(Xbyak::GetError());
        }
        kernel_ = reinterpret_cast<decltype(kernel_)>(jit_ker());
        return code;
    }

    void operator()(const jit_call_args* args) const {
        assert(kernel_);
        kernel_(args);
    }

    void operator()(const jit_call_args& args) const {
        this->operator()(&args);
    }

protected:
    jit_params params_;

private:
    kernel_func kernel_;
};

} // namespace intel_cpu
} // namespace ov
