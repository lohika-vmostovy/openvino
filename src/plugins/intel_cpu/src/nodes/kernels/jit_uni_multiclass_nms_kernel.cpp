// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "jit_uni_multiclass_nms_kernel.hpp"

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

template<cpu_isa_t isa>
jit_uni_multiclass_nms_kernel_impl<isa>::jit_uni_multiclass_nms_kernel_impl() :
    jit_uni_multiclass_nms_kernel{},
    jit_generator{jit_name()},
    reg_pool_{RegistersPool::create<isa>({Reg64(Operand::RAX), Reg64(Operand::RCX), Reg64(Operand::RBP), abi_param1})},
    reg_params_{abi_param1} {
}

template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::generate() {
    preamble();

    Label box_loop_label;
    Label box_loop_continue_label;
    Label box_loop_end_label;

    Label iou_loop_label;
    Label iou_loop_continue_label;
    Label iou_loop_end_label;

    Label const_fhalf;

    // float iou_threshold;
    PVmm reg_iou_threshold {reg_pool_};
    uni_vbroadcastss(reg_iou_threshold, ptr[reg_params_ + offsetof(jit_nms_call_args, iou_threshold)]);
    // float score_threshold, used as integer for equality checks
    PReg32 reg_score_threshold {reg_pool_};
    mov(reg_score_threshold, dword[reg_params_ + offsetof(jit_nms_call_args, score_threshold)]);
    // float nms_eta;
    PVmm reg_nms_eta {reg_pool_};
    uni_vbroadcastss(reg_nms_eta, ptr[reg_params_ + offsetof(jit_nms_call_args, nms_eta)]);
    // float coordinates_offset;
    PVmm reg_coordinates_offset {reg_pool_};
    uni_vbroadcastss(reg_coordinates_offset, ptr[reg_params_ + offsetof(jit_nms_call_args, coordinates_offset)]);

    // Box* boxes_ptr;
    PReg64 reg_boxes_ptr {reg_pool_};
    mov(reg_boxes_ptr, qword[reg_params_ + offsetof(jit_nms_call_args, boxes_ptr)]);
    // int num_boxes;
    PReg64 reg_num_boxes {reg_pool_};
    mov(static_cast<Reg64>(reg_num_boxes).cvt32(), dword[reg_params_ + offsetof(jit_nms_call_args, num_boxes)]);
    // const float* coords_ptr;
    PReg64 reg_coords_array_ptr {reg_pool_};
    mov(reg_coords_array_ptr, qword[reg_params_ + offsetof(jit_nms_call_args, coords_ptr)]);

    PVmm reg_halfs {reg_pool_};
    uni_vbroadcastss(reg_halfs, ptr[rip + const_fhalf]);

    // int num_boxes_selected = 0;
    PReg64 reg_num_boxes_selected {reg_pool_};
    xor_(reg_num_boxes_selected, reg_num_boxes_selected);

    // for (size_t i = 0; i < args->num_boxes; i++) {
    PReg64 reg_i {reg_pool_};
    xor_(reg_i, reg_i);
    L(box_loop_label);
    {
        cmp(reg_i, reg_num_boxes);
        jge(box_loop_end_label, T_NEAR);

        PReg32 box_score {reg_pool_};

        PVmm xminI {reg_pool_};
        PVmm yminI {reg_pool_};
        PVmm xmaxI {reg_pool_};
        PVmm ymaxI {reg_pool_};
        {
            // const float* box_coords_ptr = &args->coords_ptr[args->boxes_ptr[i].box_idx * 4];
            PReg64 box_ptr {reg_pool_};
            get_box_ptr(reg_boxes_ptr, reg_i, box_ptr);
            mov(box_score, dword[box_ptr + offsetof(Box, score)]);
            const PReg64& box_coords_ptr = box_ptr;
            get_box_coords_ptr(box_ptr, reg_coords_array_ptr, box_coords_ptr);

            uni_vbroadcastss(xminI, ptr[box_coords_ptr]);
            uni_vbroadcastss(yminI, ptr[box_coords_ptr + 4]);
            uni_vbroadcastss(xmaxI, ptr[box_coords_ptr + 8]);
            uni_vbroadcastss(ymaxI, ptr[box_coords_ptr + 12]);
        }

        // float areaI = (ymaxI - yminI + norm) * (xmaxI - xminI + norm);
        PVmm areaI {reg_pool_};
        {
            PVmm width {reg_pool_};
            PVmm height {reg_pool_};
            uni_vsubps(width, xmaxI, xminI);
            uni_vsubps(height, ymaxI, yminI);
            uni_vaddps(width, width, reg_coordinates_offset);
            uni_vaddps(height, height, reg_coordinates_offset);
            uni_vmulps(areaI, width, height);
        }

#undef BUG_IN_REFERENCE_IMPL_IS_FIXED
#ifdef BUG_IN_REFERENCE_IMPL_IS_FIXED
        // for (int reg_j = 0; reg_j < num_boxes_selected; reg_j += simd_width)
        PReg64 reg_j {reg_pool_};
        xor_(reg_j, reg_j);
#else
        // for (int reg_j = (box_score == reg_score_threshold) ? max((num_boxes_selected - 1), 0) : 0;
        //                   reg_j < num_boxes_selected;
        //                   reg_j += simd_width)
        PReg64 reg_j {reg_pool_};
        mov(reg_j, reg_num_boxes_selected);
        dec(reg_j);
        PReg64 reg_zero {reg_pool_};
        xor_(reg_zero, reg_zero);
        cmp(box_score, reg_score_threshold);
        cmovne(reg_j, reg_zero);
        test(reg_num_boxes_selected, reg_num_boxes_selected);
        cmovz(reg_j, reg_zero);
        reg_zero.release();
#endif
        L(iou_loop_label);
        {
            cmp(reg_j, reg_num_boxes_selected);
            jge(iou_loop_end_label, T_NEAR);

            PVmm xminJ {reg_pool_};
            PVmm yminJ {reg_pool_};
            PVmm xmaxJ {reg_pool_};
            PVmm ymaxJ {reg_pool_};

            mov(rax, qword[reg_params_ + offsetof(jit_nms_call_args, xmin_ptr)]);
            load_simd_register(xminJ, rax, reg_num_boxes_selected, reg_j);
            mov(rax, qword[reg_params_ + offsetof(jit_nms_call_args, ymin_ptr)]);
            load_simd_register(yminJ, rax, reg_num_boxes_selected, reg_j);
            mov(rax, qword[reg_params_ + offsetof(jit_nms_call_args, xmax_ptr)]);
            load_simd_register(xmaxJ, rax, reg_num_boxes_selected, reg_j);
            mov(rax, qword[reg_params_ + offsetof(jit_nms_call_args, ymax_ptr)]);
            load_simd_register(ymaxJ, rax, reg_num_boxes_selected, reg_j);

            // const float iou = intersection_over_union();
            PVmm reg_iou;
            {
                // float areaJ = (ymaxJ - yminJ + norm) * (xmaxJ - xminJ + norm);
                PVmm areaJ;
                {
                    PVmm width {reg_pool_};
                    PVmm height {reg_pool_};
                    uni_vsubps(width, xmaxJ, xminJ);
                    uni_vsubps(height, ymaxJ, yminJ);
                    uni_vaddps(width, width, reg_coordinates_offset);
                    uni_vaddps(height, height, reg_coordinates_offset);
                    uni_vmulps(width, width, height);
                    areaJ = std::move(width);
                }

                PVmm intersection_area;
                {
                    PVmm tmp0 {reg_pool_};

                    // std::max(std::min(ymaxI, ymaxJ) - std::max(yminI, yminJ) + norm, 0.f)
                    PVmm height {reg_pool_};
                    uni_vminps(height, ymaxI, ymaxJ);
                    uni_vmaxps(tmp0, yminI, yminJ);
                    uni_vsubps(height, height, tmp0);
                    uni_vaddps(height, height, reg_coordinates_offset);
                    uni_vpxor(tmp0, tmp0, tmp0);
                    uni_vmaxps(height, height, tmp0);
                    ymaxJ.release();
                    yminJ.release();

                    // std::max(std::min(xmaxI, xmaxJ) - std::max(xminI, xminJ) + norm, 0.f)
                    PVmm width {reg_pool_};
                    uni_vminps(width, xmaxI, xmaxJ);
                    uni_vmaxps(tmp0, xminI, xminJ);
                    uni_vsubps(width, width, tmp0);
                    uni_vaddps(width, width, reg_coordinates_offset);
                    uni_vpxor(tmp0, tmp0, tmp0);
                    uni_vmaxps(width, width, tmp0);
                    xmaxJ.release();
                    xminJ.release();

                    uni_vmulps(width, width, height);
                    intersection_area = std::move(width);
                }

                // iou_denominator = (areaI + areaJ - intersection_area)
                PVmm iou_denominator {reg_pool_};
                uni_vaddps(iou_denominator, areaJ, areaI);
                uni_vsubps(iou_denominator, iou_denominator, intersection_area);

                // if (areaI <= 0.f || areaJ <= 0.f)
                //     reg_iou = 0.f;
                PVmm zero {reg_pool_};
                uni_vpxor(zero, zero, zero);
                if (isa == avx512_core) {
                    vcmpps(k1, areaI, zero, VCMPPS_GT);
                    vcmpps(k2, areaJ, zero, VCMPPS_GT);
                    kandw(k1, k1, k2);
                    reg_iou = PVmm {reg_pool_};
                    vdivps(static_cast<Vmm>(reg_iou)|k1|T_z, intersection_area, iou_denominator);
                } else {
                    PVmm maskI {reg_pool_};
                    PVmm maskJ {reg_pool_};
                    uni_vcmpps(maskI, areaI, zero, VCMPPS_GT);
                    uni_vcmpps(maskJ, areaJ, zero, VCMPPS_GT);
                    uni_vandps(maskI, maskI, maskJ);
                    // reg_iou = intersection_area / (areaI + areaJ - intersection_area)
                    reg_iou = PVmm {reg_pool_};
                    uni_vdivps(reg_iou, intersection_area, iou_denominator);
                    uni_vandps(reg_iou, reg_iou, maskI);
                }
            }

            // box_is_selected = (iou < iou_threshold);
            // if (!box_is_selected)
            //     break;
            {
                if (isa == avx512_core) {
                    vcmpps(k1, reg_iou, reg_iou_threshold, VCMPPS_GE);
                    ktestw(k1, k1);
                    jnz(box_loop_continue_label, T_NEAR);
                } else {
                    uni_vcmpps(reg_iou, reg_iou, reg_iou_threshold, VCMPPS_GE);
                    uni_vtestps(reg_iou, reg_iou);
                    jnz(box_loop_continue_label, T_NEAR);
                }
            }

            // L(iou_loop_continue_label);
            add(reg_j, simd_width);
            jmp(iou_loop_label, T_NEAR);
        }
        L(iou_loop_end_label);

        // if (iou_threshold > 0.5f) {
        //     iou_threshold *= args->nms_eta;
        // }
        // args->boxes[num_boxes_selected++] = args->boxes[i];
        {
            Label copy_box_label;
            if (isa == avx512_core) {
                vcmpps(k1, reg_iou_threshold, reg_halfs, VCMPPS_GT);
                ktestw(k1, k1);
                jz(copy_box_label, T_NEAR);
            } else {
                PVmm tmp {reg_pool_};
                uni_vcmpps(tmp, reg_iou_threshold, reg_halfs, VCMPPS_GT);
                uni_vtestps(tmp, tmp);
                jz(copy_box_label, T_NEAR);
            }
            uni_vmulps(reg_iou_threshold, reg_iou_threshold, reg_nms_eta);
            L(copy_box_label);
            {
                // copy box
                {
                    PReg64 src {reg_pool_};
                    PReg64 dst {reg_pool_};
                    assert(sizeof(Box) == 16);
                    mov(src, reg_i);
                    sal(src, 4);
                    lea(src, ptr[reg_boxes_ptr + src]);
                    mov(dst, reg_num_boxes_selected);
                    sal(dst, 4);
                    lea(dst, ptr[reg_boxes_ptr + dst]);
                    mov(rax, qword[src]);
                    mov(qword[dst], rax);
                    mov(rax, qword[src + 8]);
                    mov(qword[dst + 8], rax);
                }

                // copy box coords
                {
                    mov(rax, qword[reg_params_ + offsetof(jit_nms_call_args, xmin_ptr)]);
                    uni_vmovss(dword[rax + sizeof(float)*reg_num_boxes_selected], xminI);
                    mov(rax, qword[reg_params_ + offsetof(jit_nms_call_args, ymin_ptr)]);
                    uni_vmovss(dword[rax + sizeof(float)*reg_num_boxes_selected], yminI);
                    mov(rax, qword[reg_params_ + offsetof(jit_nms_call_args, xmax_ptr)]);
                    uni_vmovss(dword[rax + sizeof(float)*reg_num_boxes_selected], xmaxI);
                    mov(rax, qword[reg_params_ + offsetof(jit_nms_call_args, ymax_ptr)]);
                    uni_vmovss(dword[rax + sizeof(float)*reg_num_boxes_selected], ymaxI);
                }

                inc(reg_num_boxes_selected);
            }
        }

        L(box_loop_continue_label);
        inc(reg_i);
        jmp(box_loop_label, T_NEAR);
    }
    L(box_loop_end_label);

    // *args->num_boxes_selected = num_boxes_selected;
    mov(rax, qword[reg_params_ + offsetof(jit_nms_call_args, num_boxes_selected_ptr)]);
    mov(qword[rax], reg_num_boxes_selected);

    postamble();

    // Constants
    std::uint32_t mem_placeholder;

    L_aligned(const_fhalf);
    {
        float vc_fhalf {0.5f};
        memcpy(&mem_placeholder, &vc_fhalf, sizeof(float));
        dd(mem_placeholder);
    }
}

template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::get_box_ptr(
    Reg64 boxes_ptr, Reg64 box_idx, Reg64 result) {
    assert(sizeof(Box) == 16);
    mov(result, box_idx);
    sal(result, 4);
    lea(result, ptr[boxes_ptr + result]);
}

template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::get_box_coords_ptr(
    Reg64 box_ptr, Reg64 coords_array_ptr, Reg64 result) {
    mov(result.cvt32(), dword[box_ptr + offsetof(Box, box_idx)]);
    sal(result, 4);
    lea(result, ptr[coords_array_ptr + result]);
}

template <cpu_isa_t isa>
void jit_uni_multiclass_nms_kernel_impl<isa>::load_simd_register(
    const Vmm& reg, const Reg64& buff_ptr, const Reg64& buff_size, const Reg64& index) {
    Reg64 num_elements {Operand::RCX};
    mov(num_elements, buff_size);
    sub(num_elements, index);
    PReg64 reg_simd_width {reg_pool_};
    mov(reg_simd_width, simd_width);
    cmp(num_elements, reg_simd_width);
    cmovg(num_elements, reg_simd_width);

    // mask = 2^num_elements - 1
    PReg64 mask {reg_pool_};
    mov(mask, 1);
    sal(mask, num_elements.cvt8());
    dec(mask);
    kmovq(k1, mask);

    vmovups(reg | k1 | T_z, ptr[buff_ptr + sizeof(float) * index]);
}


template class jit_uni_multiclass_nms_kernel_impl<sse41>;
template class jit_uni_multiclass_nms_kernel_impl<avx2>;
template class jit_uni_multiclass_nms_kernel_impl<avx512_core>;

} // namespace node
} // namespace intel_cpu
} // namespace ov
