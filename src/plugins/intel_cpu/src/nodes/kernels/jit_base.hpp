// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <cpu/x64/jit_generator.hpp>

namespace ov {
namespace intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;

class jit_base : public x64::jit_generator {
public:
    static constexpr size_t xmm_len = 16;
    static constexpr size_t ymm_len = 32;
    static constexpr size_t zmm_len = 64;

    explicit jit_base(const char *name, x64::cpu_isa_t max_cpu_isa)
        : x64::jit_generator{name, nullptr, 256 * 1024, true, max_cpu_isa}
        , max_cpu_isa_{max_cpu_isa} {}
    virtual ~jit_base() = default;

    bool is_valid_isa(const x64::cpu_isa_t isa) const {
        return x64::is_subset(isa, max_cpu_isa_) && x64::mayiuse(isa);
    }

    using x64::jit_generator::push;
    using x64::jit_generator::pop;

    void push(const Xbyak::Xmm &xmm) {
        if (xmm.isXMM()) {
            sub(rsp, xmm_len);
            uni_vmovdqu(ptr[rsp], xmm);
        } else if (xmm.isYMM()) {
            sub(rsp, ymm_len);
            uni_vmovdqu(ptr[rsp], Xbyak::Ymm{xmm.getIdx()});
        } else if (xmm.isZMM()) {
            sub(rsp, zmm_len);
            uni_vmovdqu(ptr[rsp], Xbyak::Zmm{xmm.getIdx()});
        }
    }

    void push(const std::vector<Xbyak::Xmm> &xmms) {
        sub(rsp, get_regs_size(xmms));
        size_t offset = 0;
        for (const auto& xmm : xmms) {
            if (xmm.isXMM()) {
                uni_vmovdqu(ptr[rsp + offset], xmm);
                offset += xmm_len;
            } else if (xmm.isYMM()) {
                uni_vmovdqu(ptr[rsp + offset], Xbyak::Ymm{xmm.getIdx()});
                offset += ymm_len;
            } else if (xmm.isZMM()) {
                uni_vmovdqu(ptr[rsp + offset], Xbyak::Zmm{xmm.getIdx()});
                offset += zmm_len;
            }
        }
    }

    void pop(const Xbyak::Xmm &xmm) {
        if (xmm.isXMM()) {
            uni_vmovdqu(xmm, ptr[rsp]);
            add(rsp, xmm_len);
        } else if (xmm.isYMM()) {
            uni_vmovdqu(Xbyak::Ymm{xmm.getIdx()}, ptr[rsp]);
            add(rsp, ymm_len);
        } else if (xmm.isZMM()) {
            uni_vmovdqu(Xbyak::Zmm{xmm.getIdx()}, ptr[rsp]);
            add(rsp, zmm_len);
        }
    }

    void pop(const std::vector<Xbyak::Xmm> &xmms) {
        size_t offset = 0;
        for (const auto& xmm : xmms) {
            if (xmm.isXMM()) {
                uni_vmovdqu(xmm, ptr[rsp + offset]);
                offset += xmm_len;
            } else if (xmm.isYMM()) {
                uni_vmovdqu(Xbyak::Ymm{xmm.getIdx()}, ptr[rsp + offset]);
                offset += ymm_len;
            } else if (xmm.isZMM()) {
                uni_vmovdqu(Xbyak::Zmm{xmm.getIdx()}, ptr[rsp + offset]);
                offset += zmm_len;
            }
        }
        add(rsp, offset);
    }

    void uni_vaddps(const Xbyak::Xmm &x,
                    const Xbyak::Xmm &op1,
                    const Xbyak::Operand &op2) {
        if (is_valid_isa(x64::avx)) {
            vaddps(x, op1, op2);
        } else {
            if (x.getIdx() == op1.getIdx()) {
                addps(x, op2);
            } else if (x.isEqualIfNotInherited(op2)) {
                addps(x, op1);
            } else {
                movups(x, op1);
                addps(x, op2);
            }
        }
    }

    void uni_vaddps(const Xbyak::Ymm &x,
                    const Xbyak::Ymm &op1,
                    const Xbyak::Operand &op2) {
        vaddps(x, op1, op2);
    }

    void uni_vsubps(const Xbyak::Xmm &x,
                    const Xbyak::Xmm &op1,
                    const Xbyak::Operand &op2) {
        if (is_valid_isa(x64::avx)) {
            vsubps(x, op1, op2);
        } else {
            if (x.getIdx() == op1.getIdx()) {
                subps(x, op2);
            } else if (x.isEqualIfNotInherited(op2)) {
                push(op1);
                subps(op1, op2);
                movups(x, op1);
                pop(op1);
            } else {
                movups(x, op1);
                subps(x, op2);
            }
        }
    }

    void uni_vsubps(const Xbyak::Ymm &x,
                    const Xbyak::Ymm &op1,
                    const Xbyak::Operand &op2) {
        vsubps(x, op1, op2);
    }

    static constexpr unsigned VCMPPS_LE = 0x02;
    static constexpr unsigned VCMPPS_GT = 0x0e;

    void uni_vcmpps(const Xbyak::Xmm &x,
                    const Xbyak::Xmm &op1,
                    const Xbyak::Operand &op2,
                    const int cmp_predicate) {
        if (is_valid_isa(x64::avx)) {
            vcmpps(x, op1, op2, cmp_predicate);
        } else {
            if (x.getIdx() == op1.getIdx()) {
                cmpps(x, op2, cmp_predicate);
            } else if (x.isEqualIfNotInherited(op2)) {
                push(op1);
                cmpps(op1, op2, cmp_predicate);
                movups(x, op1);
                pop(op1);
            } else {
                movups(x, op1);
                cmpps(x, op2, cmp_predicate);
            }
        }
    }

    void uni_vcmpps(const Xbyak::Ymm &x1,
                    const Xbyak::Ymm &x2,
                    const Xbyak::Operand &op,
                    const int cmp_predicate) {
        vcmpps(x1, x2, op, cmp_predicate);
    }

protected:
    x64::cpu_isa_t max_cpu_isa_;

private:
    static size_t get_regs_size(const std::vector<Xbyak::Xmm>& xmms) {
        size_t total_size = 0;
        for (const auto& xmm : xmms) {
            if (xmm.isXMM()) {
                total_size += xmm_len;
            } else if (xmm.isYMM()) {
                total_size += ymm_len;
            } else if (xmm.isZMM()) {
                total_size += zmm_len;
            }
        }
        return total_size;
    }
};

} // namespace intel_cpu
} // namespace ov
