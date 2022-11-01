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
#include "jit_base.hpp"

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

    void emu_vgatherdps(const Xbyak::Xmm &xmm_val,
                        const Xbyak::Reg64 &reg_addr,
                        const Xbyak::Xmm &xmm_index,
                        const int &scale,
                        const int &disp,
                        const Xbyak::Reg &reg_mask) {
        const size_t kDataTypeSize = sizeof(float);
        if (is_valid_isa(x64::avx512_core)) {
            assert(reg_mask.isOPMASK());
            vgatherdps(xmm_val, ptr[reg_addr + xmm_index * scale + disp]);
        } else if (is_valid_isa(x64::avx2)) {
            assert(reg_mask.isYMM());
            Xbyak::Ymm ymm_mask{reg_mask.getIdx()};
            vgatherdps(xmm_val, ptr[reg_addr + xmm_index * scale + disp], ymm_mask);
        } else {
            const size_t kSimdWidth = x64::cpu_isa_traits<x64::sse41>::vlen / kDataTypeSize;
            assert(reg_mask.isXMM());
            Xbyak::Xmm xmm_mask{reg_mask.getIdx()};
            assert(xmm_val.getKind() == xmm_index.getKind());
            assert(xmm_index.getKind() == xmm_mask.getKind());

            std::vector<Xbyak::Reg> not_available_reg{reg_addr};
            const Xbyak::Reg64 idx = register_pool()->getInplaceFree<Xbyak::Reg64>(not_available_reg);
            const Xbyak::Reg64 mask = register_pool()->getInplaceFree<Xbyak::Reg64>(not_available_reg);

            push(idx);
            push(mask);
            xor_(idx, idx);
            xor_(mask, mask);

            for (int i = 0; i < static_cast<int>(kSimdWidth); i++) {
                Xbyak::Label gather_end;
                uni_vpextrd(mask, xmm_mask, i);
                cmp(mask, 0xFFFFFFFF);
                jne(gather_end, T_NEAR);
                uni_vpextrd(idx, xmm_index, i);
                Xbyak::Address addr = ptr[reg_addr + idx * scale + disp];
                uni_vpinsrd(xmm_val, xmm_val, addr, i);
                L(gather_end);
            }

            pop(mask);
            pop(idx);
        }
    }

    void emu_vscatterdps(const Xbyak::Reg64& reg_addr,
                         const Xbyak::Xmm& xmm_index,
                         const int scale,
                         const int disp,
                         const Xbyak::Xmm& xmm_val,
                         const Xbyak::Reg& reg_mask) {
        const size_t kDataTypeSize = sizeof(float);
        if (is_valid_isa(x64::avx512_core)) {
            assert(reg_mask.isOPMASK());
            vscatterdps(ptr[reg_addr + xmm_index * scale + disp], xmm_val);
        } else {
            assert(reg_mask.isXMM() || reg_mask.isYMM());
            const size_t kXmmSimdWidth = x64::cpu_isa_traits<x64::sse41>::vlen / kDataTypeSize;
            const size_t kYmmSimdWidth = x64::cpu_isa_traits<x64::avx2>::vlen / kDataTypeSize;
            Xbyak::Xmm xmm_mask{reg_mask.getIdx(), reg_mask.getKind(), static_cast<int>(reg_mask.getBit())};
            assert(xmm_val.getKind() == xmm_index.getKind());
            assert(xmm_index.getKind() == xmm_mask.getKind());

            std::vector<Xbyak::Reg> not_available_reg{reg_addr};
            std::vector<Xbyak::Xmm> not_available_xmm{xmm_index, xmm_val, xmm_mask};
            const Xbyak::Reg64 idx = register_pool()->getInplaceFree<Xbyak::Reg64>(not_available_reg);
            const Xbyak::Reg64 mask = register_pool()->getInplaceFree<Xbyak::Reg64>(not_available_reg);
            const Xbyak::Reg64 val = register_pool()->getInplaceFree<Xbyak::Reg64>(not_available_reg);
            const Xbyak::Xmm xmm_mask_temp = register_pool()->getInplaceFree<Xbyak::Xmm>(not_available_xmm);
            const Xbyak::Xmm xmm_index_temp = register_pool()->getInplaceFree<Xbyak::Xmm>(not_available_xmm);
            const Xbyak::Xmm xmm_val_temp = register_pool()->getInplaceFree<Xbyak::Xmm>(not_available_xmm);

            push(idx);
            push(mask);
            push(val);
            if (is_valid_isa(x64::avx2)) {
                push(Xbyak::Ymm{xmm_mask_temp.getIdx()});
                push(Xbyak::Ymm{xmm_index_temp.getIdx()});
                push(Xbyak::Ymm{xmm_val_temp.getIdx()});
            }
            xor_(idx, idx);
            xor_(mask, mask);
            xor_(val, val);

            auto store_xmm = [&](const Xbyak::Xmm& xmm_mask,
                                 const Xbyak::Xmm& xmm_index,
                                 const Xbyak::Xmm& xmm_val) {
                for (int i = 0; i < static_cast<int>(kXmmSimdWidth); i++) {
                    Xbyak::Label scatter_end;
                    uni_vpextrd(mask, xmm_mask, i);
                    cmp(mask, 0xFFFFFFFF);
                    jne(scatter_end, T_NEAR);
                    uni_vpextrd(idx, xmm_index, i);
                    Xbyak::Address addr = ptr[reg_addr + idx * scale];
                    uni_vpextrd(val, xmm_val, i);
                    mov(addr, val);
                    L(scatter_end);
                }
            };

            if (is_valid_isa(x64::avx2)) {
                for (int i = 0; i < static_cast<int>(kYmmSimdWidth / kXmmSimdWidth); i++) {
                    vextracti128(xmm_mask_temp, Xbyak::Ymm{xmm_mask.getIdx()}, i);
                    vextracti128(xmm_index_temp, Xbyak::Ymm{xmm_index.getIdx()}, i);
                    vextracti128(xmm_val_temp, Xbyak::Ymm{xmm_val.getIdx()}, i);
                    store_xmm(xmm_mask_temp, xmm_index_temp, xmm_val_temp);
                }
            } else {
                store_xmm(xmm_mask, xmm_index, xmm_val);
            }

            if (is_valid_isa(x64::avx2)) {
                pop(Xbyak::Ymm{xmm_val_temp.getIdx()});
                pop(Xbyak::Ymm{xmm_index_temp.getIdx()});
                pop(Xbyak::Ymm{xmm_mask_temp.getIdx()});
            }
            pop(val);
            pop(mask);
            pop(idx);
        }
    }

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
