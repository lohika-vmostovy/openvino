// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <dnnl_types.h>
#include "ie_common.h"
#include "utils/cpu_utils.hpp"
#include <utility>
#include "stack_allocator.hpp"

namespace ov {
namespace intel_cpu {

namespace x64 = dnnl::impl::cpu::x64;

/**
 * The RegistersPool is the base class for the IsaRegistersPool template:
 *      template <x64::cpu_isa_t isa>
 *      class IsaRegistersPool : public RegistersPool;
 *
 * The registers pool must be created by instantiating the IsaRegistersPool template, like the next:
 *      RegistersPool::Ptr regPool = RegistersPool::create<isa>({
 *          // the list of the registers to be excluded from pool
 *          Reg64(Operand::RAX), Reg64(Operand::RCX), Reg64(Operand::RDX), Reg64(Operand::RBX),
 *          Reg64(Operand::RSP), Reg64(Operand::RBP), Reg64(Operand::RSI), Reg64(Operand::RDI)
 *      });
 */
class RegistersPool {
public:
    using Ptr = std::shared_ptr<RegistersPool>;
    static constexpr int anyIdx = -1;

    /**
     * The scoped wrapper for the Xbyak registers.
     * By creating it you are getting the register from the pool RegistersPool.
     * It could be created by using constructor with RegistersPool as an argument, like the next:
     *      const RegistersPool::Reg<Xbyak::Reg64> reg {regPool};
     * The destructor will return the register to the pool. Or it could be returned manually:
     *      reg.release();
     * @tparam TReg Xbyak register class
     */
    template<typename TReg>
    class Reg {
        friend class RegistersPool;
    public:
        Reg() {}
        Reg(const RegistersPool::Ptr& regPool) { initialize(regPool); }
        Reg(const RegistersPool::Ptr& regPool, int requestedIdx) { initialize(regPool, requestedIdx); }
        ~Reg() { release(); }
        Reg(Reg&& other)  noexcept {
            this->operator=(std::move(other));
        }
        Reg& operator=(Reg&& other)  noexcept {
            release();
            reg = other.reg;
            regPool = std::move(other.regPool);
            return *this;
        }
        operator TReg&() { ensureValid(); return reg; }
        operator const TReg&() const { ensureValid(); return reg; }
        operator Xbyak::RegExp() const { ensureValid(); return reg; }
        Reg& operator=(const StackAllocator::Address& addr) {
            stack_mov(*this, addr);
            return *this;
        }
        int getIdx() const { ensureValid(); return reg.getIdx(); }
        friend Xbyak::RegExp operator+(const Reg& lhs, const Xbyak::RegExp& rhs) {
            lhs.ensureValid();
            return lhs.operator Xbyak::RegExp() + rhs;
        }
        void release() {
            if (regPool) {
                regPool->returnToPool(reg);
                regPool.reset();
            }
        }
        bool isInitialized() const { return static_cast<bool>(regPool); }

    private:
        void ensureValid() const {
            if (!isInitialized()) {
                IE_THROW() << "RegistersPool::Reg is either not initialized or released";
            }
        }

        void initialize(const RegistersPool::Ptr& pool, int requestedIdx = anyIdx) {
            static_assert(is_any_of<TReg, Xbyak::Xmm, Xbyak::Ymm, Xbyak::Zmm,
                                  Xbyak::Reg8, Xbyak::Reg16, Xbyak::Reg32, Xbyak::Reg64, Xbyak::Opmask>::value,
                          "Unsupported TReg by RegistersPool::Reg. Please, use the following Xbyak registers either "
                          "Reg8, Reg16, Reg32, Reg64, Xmm, Ymm, Zmm or Opmask");
            release();
            reg = TReg(pool->template getFree<TReg>(requestedIdx));
            regPool = pool;
        }

    private:
        TReg reg;
        RegistersPool::Ptr regPool;
    };

    virtual ~RegistersPool() {
        checkUniqueAndUpdate(false);
    }

    template <x64::cpu_isa_t isa>
    static Ptr create(std::initializer_list<Xbyak::Reg> regsToExclude);
    static Ptr create(x64::cpu_isa_t isa);
    static Ptr create(x64::cpu_isa_t isa, std::initializer_list<Xbyak::Reg> regsToExclude);

    template<typename TReg>
    size_t countFree() const {
        static_assert(is_any_of<TReg, Xbyak::Xmm, Xbyak::Ymm, Xbyak::Zmm,
                              Xbyak::Reg8, Xbyak::Reg16, Xbyak::Reg32, Xbyak::Reg64, Xbyak::Opmask>::value,
                      "Unsupported TReg by RegistersPool::Reg. Please, use the following Xbyak registers either "
                      "Reg8, Reg16, Reg32, Reg64, Xmm, Ymm, Zmm or Opmask");
        if (std::is_base_of<Xbyak::Mmx, TReg>::value) {
            return simdSet.countUnused();
        } else if (std::is_same<TReg, Xbyak::Reg8>::value || std::is_same<TReg, Xbyak::Reg16>::value ||
                   std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Reg64>::value) {
            return generalSet.countUnused();
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            return countUnusedOpmask();
        }
    }

    template<typename TReg>
    inline TReg getInplaceFree(std::vector<Xbyak::Reg>& not_available) const {
        static_assert(std::is_base_of<Xbyak::Reg, TReg>::value, "Xbyak::Reg should be base of TReg !!");
        PhysicalSet localGeneralSet{16};
        localGeneralSet.exclude(Xbyak::Reg64(Xbyak::Operand::RSP));
        for (const auto& reg : not_available) {
            localGeneralSet.exclude(Xbyak::Reg64(reg.getIdx()));
        }
        const auto reg = TReg{localGeneralSet.getUnused(anyIdx)};
        not_available.push_back(reg);
        return std::move(reg);
    }

    template<typename TVmm>
    inline TVmm getInplaceFree(std::vector<Xbyak::Xmm>& not_available) const {
        static_assert(std::is_base_of<Xbyak::Xmm, TVmm>::value, "Xbyak::Xmm should be base of TVmm !!");
        PhysicalSet localSimdSet{simdRegistersNumber};
        for (const auto& xmm : not_available) {
            localSimdSet.exclude(Xbyak::Xmm(xmm.getIdx()));
        }
        const auto reg = TVmm{localSimdSet.getUnused(anyIdx)};
        not_available.push_back(reg);
        return std::move(reg);
    }

protected:
    class PhysicalSet {
    public:
        PhysicalSet(int size) : isFreeIndexVector(size, true) {}

        void setAsUsed(int regIdx) {
            if (regIdx >= isFreeIndexVector.size() || regIdx < 0) {
                IE_THROW() << "regIdx is out of bounds in RegistersPool::PhysicalSet::setAsUsed()";
            }
            if (!isFreeIndexVector[regIdx]) {
                IE_THROW() << "Inconsistency in RegistersPool::PhysicalSet::setAsUsed()";
            }
            isFreeIndexVector[regIdx] = false;
        }

        void setAsUnused(int regIdx) {
            if (regIdx >= isFreeIndexVector.size() || regIdx < 0) {
                IE_THROW() << "regIdx is out of bounds in RegistersPool::PhysicalSet::setAsUsed()";
            }
            if (isFreeIndexVector[regIdx]) {
                IE_THROW() << "Inconsistency in RegistersPool::PhysicalSet::setAsUnused()";
            }
            isFreeIndexVector[regIdx] = true;
        }

        int getUnused(int requestedIdx) {
            if (requestedIdx == anyIdx) {
                return getFirstFreeIndex();
            } else {
                if (requestedIdx >= isFreeIndexVector.size() || requestedIdx < 0) {
                    IE_THROW() << "requestedIdx is out of bounds in RegistersPool::PhysicalSet::getUnused()";
                }
                if (!isFreeIndexVector[requestedIdx]) {
                    IE_THROW() << "The register with index #" << requestedIdx << " already used in the RegistersPool";
                }
                return requestedIdx;
            }
        }

        void exclude(Xbyak::Reg reg) {
            isFreeIndexVector.at(reg.getIdx()) = false;
        }

        size_t countUnused() const {
            size_t count = 0;
            for (const auto& isFree : isFreeIndexVector) {
                if (isFree) {
                    ++count;
                }
            }
            return count;
        }

    private:
        int getFirstFreeIndex() {
            for (int c = 0; c < isFreeIndexVector.size(); ++c) {
                if (isFreeIndexVector[c]) {
                    return c;
                }
            }
            IE_THROW() << "Not enough registers in the RegistersPool";
        }

    private:
        std::vector<bool> isFreeIndexVector;
    };

    virtual int getFreeOpmask(int requestedIdx) { IE_THROW() << "getFreeOpmask: The Opmask is not supported in current instruction set"; }
    virtual void returnOpmaskToPool(int idx) { IE_THROW() << "returnOpmaskToPool: The Opmask is not supported in current instruction set"; }
    virtual size_t countUnusedOpmask() const { IE_THROW() << "countUnusedOpmask: The Opmask is not supported in current instruction set"; }

    RegistersPool(int simdRegistersNumber)
            : simdSet(simdRegistersNumber) {
        checkUniqueAndUpdate();
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RAX));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RCX));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RDX));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RBX));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RSP));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RBP));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RSI));
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RDI));
    }

    RegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude, int simdRegistersNumber)
            : simdSet(simdRegistersNumber) {
        checkUniqueAndUpdate();
        for (auto& reg : regsToExclude) {
            if (reg.isXMM() || reg.isYMM() || reg.isZMM()) {
                simdSet.exclude(reg);
            } else if (reg.isREG()) {
                generalSet.exclude(reg);
            }
        }
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RSP));
    }

private:
    template<typename TReg>
    int getFree(int requestedIdx) {
        if (std::is_base_of<Xbyak::Mmx, TReg>::value) {
            auto idx = simdSet.getUnused(requestedIdx);
            simdSet.setAsUsed(idx);
            return idx;
        } else if (std::is_same<TReg, Xbyak::Reg8>::value || std::is_same<TReg, Xbyak::Reg16>::value ||
                   std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Reg64>::value) {
            auto idx = generalSet.getUnused(requestedIdx);
            generalSet.setAsUsed(idx);
            return idx;
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            return getFreeOpmask(requestedIdx);
        }
    }

    template<typename TReg>
    void returnToPool(const TReg& reg) {
        if (std::is_base_of<Xbyak::Mmx, TReg>::value) {
            simdSet.setAsUnused(reg.getIdx());
        } else if (std::is_same<TReg, Xbyak::Reg8>::value || std::is_same<TReg, Xbyak::Reg16>::value ||
                   std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Reg64>::value) {
            generalSet.setAsUnused(reg.getIdx());
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            returnOpmaskToPool(reg.getIdx());
        }
    }

    void checkUniqueAndUpdate(bool isCtor = true) {
        static thread_local bool isCreated = false;
        if (isCtor) {
            if (isCreated) {
                IE_THROW() << "There should be only one instance of RegistersPool per thread";
            }
            isCreated = true;
        } else {
            isCreated = false;
        }
    }

    PhysicalSet generalSet {16};
    PhysicalSet simdSet;
    int simdRegistersNumber;
};

template <x64::cpu_isa_t isa>
class IsaRegistersPool : public RegistersPool {
public:
    IsaRegistersPool() : RegistersPool(x64::cpu_isa_traits<isa>::n_vregs) {}
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude) : RegistersPool(regsToExclude, x64::cpu_isa_traits<isa>::n_vregs) {}
};

template <>
class IsaRegistersPool<x64::avx512_core> : public RegistersPool {
public:
    IsaRegistersPool() : RegistersPool(x64::cpu_isa_traits<x64::avx512_core>::n_vregs) {
        opmaskSet.exclude(Xbyak::Opmask(0)); // the Opmask(0) has special meaning for some instructions, like gather instruction
    }

    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude)
            : RegistersPool(regsToExclude, x64::cpu_isa_traits<x64::avx512_core>::n_vregs) {
        for (auto& reg : regsToExclude) {
            if (reg.isOPMASK()) {
                opmaskSet.exclude(reg);
            }
        }
    }

    int getFreeOpmask(int requestedIdx) override {
        auto idx = opmaskSet.getUnused(requestedIdx);
        opmaskSet.setAsUsed(idx);
        return idx;
    }

    void returnOpmaskToPool(int idx) override {
        opmaskSet.setAsUnused(idx);
    }

    size_t countUnusedOpmask() const override {
        return opmaskSet.countUnused();
    }

protected:
    PhysicalSet opmaskSet {8};
};

template <>
class IsaRegistersPool<x64::avx512_core_vnni> : public IsaRegistersPool<x64::avx512_core> {
public:
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude) : IsaRegistersPool<x64::avx512_core>(regsToExclude) {}
    IsaRegistersPool() : IsaRegistersPool<x64::avx512_core>() {}
};

template <>
class IsaRegistersPool<x64::avx512_core_bf16> : public IsaRegistersPool<x64::avx512_core> {
public:
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude) : IsaRegistersPool<x64::avx512_core>(regsToExclude) {}
    IsaRegistersPool() : IsaRegistersPool<x64::avx512_core>() {}
};

template <x64::cpu_isa_t isa>
RegistersPool::Ptr RegistersPool::create(std::initializer_list<Xbyak::Reg> regsToExclude) {
    return std::make_shared<IsaRegistersPool<isa>>(regsToExclude);
}

inline
RegistersPool::Ptr RegistersPool::create(x64::cpu_isa_t isa) {
#define ISA_SWITCH_CASE(isa) case isa: return std::make_shared<IsaRegistersPool<isa>>();
        switch (isa) {
            ISA_SWITCH_CASE(x64::sse41)
            ISA_SWITCH_CASE(x64::avx)
            ISA_SWITCH_CASE(x64::avx2)
            ISA_SWITCH_CASE(x64::avx2_vnni)
            ISA_SWITCH_CASE(x64::avx512_core)
            ISA_SWITCH_CASE(x64::avx512_core_vnni)
            ISA_SWITCH_CASE(x64::avx512_core_bf16)
            case x64::avx_vnni: return std::make_shared<IsaRegistersPool<x64::avx>>();
            case x64::avx512_core_bf16_ymm: return std::make_shared<IsaRegistersPool<x64::avx512_core>>();
            case x64::avx512_core_bf16_amx_int8: return std::make_shared<IsaRegistersPool<x64::avx512_core>>();
            case x64::avx512_core_bf16_amx_bf16: return std::make_shared<IsaRegistersPool<x64::avx512_core>>();
            case x64::avx512_core_amx: return std::make_shared<IsaRegistersPool<x64::avx512_core>>();
            case x64::avx512_vpopcnt: return std::make_shared<IsaRegistersPool<x64::avx512_core>>();
            case x64::isa_any:
            case x64::amx_tile:
            case x64::amx_int8:
            case x64::amx_bf16:
            case x64::isa_all:
                IE_THROW() << "Invalid isa argument in RegistersPool::create()";
        }
        IE_THROW() << "Invalid isa argument in RegistersPool::create()";
#undef ISA_SWITCH_CASE
}

inline
RegistersPool::Ptr RegistersPool::create(x64::cpu_isa_t isa, std::initializer_list<Xbyak::Reg> regsToExclude) {
#define ISA_SWITCH_CASE(isa) case isa: return std::make_shared<IsaRegistersPool<isa>>(regsToExclude);
    switch (isa) {
        ISA_SWITCH_CASE(x64::sse41)
        ISA_SWITCH_CASE(x64::avx)
        ISA_SWITCH_CASE(x64::avx2)
        ISA_SWITCH_CASE(x64::avx2_vnni)
        ISA_SWITCH_CASE(x64::avx512_core)
        ISA_SWITCH_CASE(x64::avx512_core_vnni)
        ISA_SWITCH_CASE(x64::avx512_core_bf16)
        ISA_SWITCH_CASE(x64::avx512_core_fp16)
        case x64::avx_vnni: return std::make_shared<IsaRegistersPool<x64::avx>>(regsToExclude);
        case x64::avx512_core_bf16_ymm: return std::make_shared<IsaRegistersPool<x64::avx512_core>>(regsToExclude);
        case x64::avx512_core_bf16_amx_int8: return std::make_shared<IsaRegistersPool<x64::avx512_core>>(regsToExclude);
        case x64::avx512_core_bf16_amx_bf16: return std::make_shared<IsaRegistersPool<x64::avx512_core>>(regsToExclude);
        case x64::avx512_core_amx: return std::make_shared<IsaRegistersPool<x64::avx512_core>>(regsToExclude);
        case x64::avx512_vpopcnt: return std::make_shared<IsaRegistersPool<x64::avx512_core>>(regsToExclude);
        case x64::isa_any:
        case x64::amx_tile:
        case x64::amx_int8:
        case x64::amx_bf16:
        case x64::isa_all:
            IE_THROW() << "Invalid isa argument in RegistersPool::create()";
    }
    IE_THROW() << "Invalid isa argument in RegistersPool::create()";
#undef ISA_SWITCH_CASE
}

}   // namespace intel_cpu
}   // namespace ov
