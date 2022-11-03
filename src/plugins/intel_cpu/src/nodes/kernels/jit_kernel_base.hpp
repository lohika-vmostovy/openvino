// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include "jit_base.hpp"
#include "emitters/jit_emitter.hpp"
#include "registers_pool.hpp"
#include "stack_allocator.hpp"

namespace ov {
namespace intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;

class jit_kernel_base : public jit_base {
public:
    using jit_base::jit_base;

    void generate() override {
        this->preamble();

        registers_pool_ = RegistersPool::create(max_cpu_isa_);
        stack_allocator_ = std::unique_ptr<StackAllocator>(new StackAllocator{*this});

        generate_impl();

        registers_pool_.reset();
        stack_allocator_.reset();

        this->postamble();

        for (auto& record : emitters_map_) {
            record.second->emit_data();
        }
    }

    virtual void generate_impl() = 0;

protected:
    RegistersPool::Ptr register_pool() {
        return registers_pool_;
    }

    StackAllocator& stack_allocator() {
        if (!stack_allocator_) {
            IE_THROW() << "StackAllocator is not available in this context";
        }
        return *stack_allocator_;
    }

private:
    RegistersPool::Ptr registers_pool_;
    std::unique_ptr<StackAllocator> stack_allocator_;
    std::unordered_map<std::string, std::shared_ptr<jit_emitter>> emitters_map_;
};

} // namespace intel_cpu
} // namespace ov
