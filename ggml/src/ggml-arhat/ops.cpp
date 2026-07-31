/* 
* MIT License
*
* Copyright (c) 2026 FRAGATA COMPUTER SYSTEMS AG
* Copyright (c) 2023-2026 The ggml authors
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <cstdio>
#include <cstdint>
#include <cassert>
#include <vector>
#include <array>
#include <memory>

#include "ggml-impl.h"
#include "ggml-backend.h" 
#include "ggml-backend-impl.h" 

#include "arhat/core/runtime.hpp"

#include "ggml-arhat/ops.hpp"

namespace core = arhat::core;

constexpr bool ENABLE_LOG_UNSUPPORED_OP = false;

#define ENABLE_GATED_DELTA_NET

namespace {

//
//    Utility functions
//

void log_unsupported_op(const ggml_tensor *tensor, const char *note) {
    if (!ENABLE_LOG_UNSUPPORED_OP) {
        return;
    }
    printf("[Arhat] Unsupported op (%s) [%s] [%s]\n", 
        note, ggml_op_name(tensor->op), ggml_type_name(tensor->type));
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const ggml_tensor *src = tensor->src[i];
        if (src != nullptr) {
            printf("  [%d] [%s] [%s]\n", 
                i, ggml_op_name(src->op), ggml_type_name(src->type));
        }
    }
}

core::DataType map_data_type(ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32:
        return core::DataType::F32;
    case GGML_TYPE_F16:
        return core::DataType::F16;
    case GGML_TYPE_I8:
        return core::DataType::I8;
    case GGML_TYPE_I16:
        return core::DataType::I16;
    case GGML_TYPE_I32:
        return core::DataType::I32;
    case GGML_TYPE_I64:
        return core::DataType::I64;
    case GGML_TYPE_F64:
        return core::DataType::F64;
    case GGML_TYPE_BF16:
        return core::DataType::BF16;
    case GGML_TYPE_Q2_K:
        return core::DataType::Q2_K;
    case GGML_TYPE_Q3_K:
        return core::DataType::Q3_K;
    case GGML_TYPE_Q4_0:
        return core::DataType::Q4_0;
    case GGML_TYPE_Q4_1:
        return core::DataType::Q4_1;
    case GGML_TYPE_Q4_K:
        return core::DataType::Q4_K;
    case GGML_TYPE_Q5_0:
        return core::DataType::Q5_0;
    case GGML_TYPE_Q5_1:
        return core::DataType::Q5_1;
    case GGML_TYPE_Q5_K:
        return core::DataType::Q5_K;
    case GGML_TYPE_Q6_K:
        return core::DataType::Q6_K;
    case GGML_TYPE_Q8_0:
        return core::DataType::Q8_0;
    case GGML_TYPE_Q8_1:
        return core::DataType::Q8_1;
    case GGML_TYPE_MXFP4:
        return core::DataType::MXFP4;
    default:
        core::Error("Unsupported GGML type %d", int(type)); 
        return core::DataType(0);
    }
}

core::Prec map_prec(ggml_prec prec) {
    switch (prec) {
    case GGML_PREC_DEFAULT:
        return core::Prec::Default;
    case GGML_PREC_F32:
        return core::Prec::F32;
    default:
        GGML_ASSERT(false);
        return core::Prec(0);
    }
}

core::RopeMode map_rope_mode(int mode) {
    switch (mode) {
    case GGML_ROPE_TYPE_NORMAL:
        return core::RopeMode::Normal;
    case GGML_ROPE_TYPE_NEOX:
        return core::RopeMode::Neox;
    case GGML_ROPE_TYPE_MROPE:
        return core::RopeMode::Mrope;
    case GGML_ROPE_TYPE_VISION:
        return core::RopeMode::Vision;
    case GGML_ROPE_TYPE_IMROPE:
        return core::RopeMode::Imrope;
    default:
        GGML_ASSERT(false);
        return core::RopeMode(0);
    }
}

core::PoolOp map_pool_op(ggml_op_pool op) {
    switch (op) {
    case GGML_OP_POOL_MAX:
        return core::PoolOp::Max;
    case GGML_OP_POOL_AVG:
        return core::PoolOp::Avg;
    default:
        GGML_ASSERT(false);
        return core::PoolOp(0);
    }
}

core::ScaleMode map_scale_mode(ggml_scale_mode mode) {
    switch (mode) {
    case GGML_SCALE_MODE_NEAREST:
        return core::ScaleMode::Nearest;
    case GGML_SCALE_MODE_BILINEAR:
        return core::ScaleMode::Bilinear;
    case GGML_SCALE_MODE_BICUBIC:
        return core::ScaleMode::Bicubic;
    default:
        GGML_ASSERT(false);
        return core::ScaleMode(0);
    }
}

core::SortOrder map_sort_order(ggml_sort_order order) {
    switch (order) {
    case GGML_SORT_ORDER_ASC:
        return core::SortOrder::Asc;
    case GGML_SORT_ORDER_DESC: 
        return core::SortOrder::Desc;
    default:
        GGML_ASSERT(false);
        return core::SortOrder(0);
    }
}

core::Node *get_input(ggml_tensor *tensor, int index) {
    GGML_ASSERT(index >= 0 && index < GGML_MAX_SRC);
    ggml_tensor *src = tensor->src[index];
    if (src == nullptr) {
        core::Error("Missing input %d", index);
    }
    GGML_ASSERT(src->extra != nullptr);
    return static_cast<core::Node *>(src->extra);
}

core::Node *get_input_opt(ggml_tensor *tensor, int index) {
    GGML_ASSERT(index >= 0 && index < GGML_MAX_SRC);
    ggml_tensor *src = tensor->src[index];
    if (src == nullptr) {
        return nullptr;
    }
    GGML_ASSERT(src->extra != nullptr);
    return static_cast<core::Node *>(src->extra);
}

bool is_inplace(ggml_tensor *tensor, int index) {
    GGML_ASSERT(index >= 0 && index < GGML_MAX_SRC);
    ggml_tensor *view_src = tensor->view_src;
    if (view_src == nullptr) {
        return false;
    }
    ggml_tensor *src = tensor->src[index];
    if (src == nullptr) {
        return false;
    }
    return (view_src == src || view_src == src->view_src);
}

int get_param_i32(ggml_tensor *tensor, int index) {
    GGML_ASSERT(index >= 0 && index < GGML_MAX_OP_PARAMS / sizeof(int32_t));
    return int(tensor->op_params[index]);
}

float get_param_f32(ggml_tensor *tensor, int index) {
    GGML_ASSERT(index >= 0 && index < GGML_MAX_OP_PARAMS / sizeof(int32_t));
    static_assert(sizeof(float) == sizeof(int32_t));
    return *reinterpret_cast<float *>(&tensor->op_params[index]);
}

core::Dims get_shape(ggml_tensor *tensor) {
    int64_t *ne = tensor->ne;
    return core::Dims{int(ne[0]), int(ne[1]), int(ne[2]), int(ne[3])};
}

core::Dims get_stride(ggml_type type, size_t nb1, size_t nb2, size_t nb3) {
    size_t type_size = ggml_type_size(type);
    GGML_ASSERT(nb1 % type_size == 0);
    GGML_ASSERT(nb2 % type_size == 0);
    GGML_ASSERT(nb3 % type_size == 0);
    return core::Dims{1, int(nb1 / type_size), int(nb2 / type_size), int(nb3 / type_size)};
}

int scale_offset(ggml_type type, size_t offset) {
    size_t type_size = ggml_type_size(type);
    GGML_ASSERT(offset % type_size == 0);
    return int(offset / type_size);
}

int get_offset(ggml_tensor *tensor, int index) {
    size_t offset = size_t(get_param_i32(tensor, index));
    return scale_offset(tensor->type, offset);
}

std::array<int, core::MropeSections> get_mrope_sections(ggml_tensor *tensor, int index) {
    std::array<int, core::MropeSections> sections;
    for (int i = 0; i < core::MropeSections; i++) {
        sections[i] = get_param_i32(tensor, index + i);
    }
    return sections;
}

int64_t get_volume(const int64_t *ne) {
    int64_t v = 1;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        v *= ne[i];
    }
    return v;
}

bool is_dense(const int64_t *ne, const size_t *nb) {
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (int64_t(nb[i]) != int64_t(nb[i - 1]) * ne[i - 1]) {
            return false;
        }
    }
    return true;
}

bool view_is_reshape(const ggml_tensor *tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_VIEW);
    ggml_tensor *src0 = tensor->src[0];
    if (get_volume(src0->ne) != get_volume(tensor->ne)) {
        return false;
    }
    if (!core::CanReshape(src0->ne, src0->nb, tensor->ne, tensor->nb)) {
        return false;
    }
    size_t param0 = 0;
    memcpy(&param0, tensor->op_params, sizeof(param0));
    if (param0 != 0) {
        return false;
    }
    return true;
}

bool match_view_norm_strides(const ggml_tensor *tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_VIEW);
    ggml_tensor *src0 = tensor->src[0];
    size_t src_norm[GGML_MAX_DIMS];
    int src_rank = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (src0->ne[i] != 1) {
            src_norm[src_rank] = src0->nb[i];
            src_rank++;
        }
    }
    size_t dst_norm[GGML_MAX_DIMS];
    int dst_rank = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1) {
            dst_norm[dst_rank] = tensor->nb[i];
            dst_rank++;
        }
    }
    if (src_rank != dst_rank) {
        return false;
    }
    for (int i = 0; i < src_rank; i++) {
        if (src_norm[i] != dst_norm[i]) {
            return false;
        }
    }
    return true;
}

//
//    Node factories
//

std::unique_ptr<core::Node> CreateTensor(core::Context *context, ggml_tensor *tensor) {
    core::DataType type = map_data_type(tensor->type);
    core::Dims shape = get_shape(tensor);
    return context->CreateTensor(type, shape);
}

std::unique_ptr<core::Node> CreateDup(core::Context *context, ggml_tensor *tensor) {
    // inplace: result will reuse type and buffer of 'b'
    // otherwise: result will own new buffer with type and shape of 'a'
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateDup(a, inplace);
}

std::unique_ptr<core::Node> CreateAdd(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    core::DataType dstType = map_data_type(tensor->type);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateAdd(a, b, dstType, inplace);
}

std::unique_ptr<core::Node> CreateAddId(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    core::Node *ids = get_input(tensor, 2);
    return context->CreateAddId(a, b, ids);
}

std::unique_ptr<core::Node> CreateAdd1(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateAdd1(a, b, inplace);
}

std::unique_ptr<core::Node> CreateAcc(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    size_t nb1 = size_t(get_param_i32(tensor, 0));
    size_t nb2 = size_t(get_param_i32(tensor, 1));
    size_t nb3 = size_t(get_param_i32(tensor, 2));
    core::Dims stride = get_stride(tensor->type, nb1, nb2, nb3);
    int offset = get_offset(tensor, 4);
    bool inplace = bool(get_param_i32(tensor, 4));
    return context->CreateAcc(a, b, stride, offset, inplace);
}

std::unique_ptr<core::Node> CreateSub(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateSub(a, b, inplace);
}

std::unique_ptr<core::Node> CreateMul(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateMul(a, b, inplace);
}

std::unique_ptr<core::Node> CreateDiv(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateDiv(a, b, inplace);
}

std::unique_ptr<core::Node> CreateSqr(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateSqr(a, inplace);
}

std::unique_ptr<core::Node> CreateSqrt(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateSqrt(a, inplace);
}

std::unique_ptr<core::Node> CreateLog(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateLog(a, inplace);
}

std::unique_ptr<core::Node> CreateSin(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateSin(a, inplace);
}

std::unique_ptr<core::Node> CreateCos(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateCos(a, inplace);
}

std::unique_ptr<core::Node> CreateSum(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    return context->CreateSum(a);
}

std::unique_ptr<core::Node> CreateSumRows(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    return context->CreateSumRows(a);
}

std::unique_ptr<core::Node> CreateCumSum(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    return context->CreateCumSum(a);
}

std::unique_ptr<core::Node> CreateMean(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    return context->CreateMean(a);
}

std::unique_ptr<core::Node> CreateArgmax(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    return context->CreateArgmax(a);
}

std::unique_ptr<core::Node> CreateCountEqual(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    return context->CreateCountEqual(a, b);
}

std::unique_ptr<core::Node> CreateRepeat(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Dims shape = get_shape(tensor);
    return context->CreateRepeat(a, shape);
}

std::unique_ptr<core::Node> CreateRepeatBack(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Dims shape = get_shape(tensor);
    return context->CreateRepeatBack(a, shape);
}

std::unique_ptr<core::Node> CreateConcat(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    int dim = get_param_i32(tensor, 0);
    return context->CreateConcat(a, b, dim);
}

std::unique_ptr<core::Node> CreateSiluBack(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    return context->CreateSiluBack(a, b);
}

std::unique_ptr<core::Node> CreateNorm(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    float eps = get_param_f32(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateNorm(a, eps, inplace);
}

std::unique_ptr<core::Node> CreateRmsNorm(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    float eps = get_param_f32(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateRmsNorm(a, eps, inplace);
}

std::unique_ptr<core::Node> CreateRmsNormBack(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    float eps = get_param_f32(tensor, 0);
    return context->CreateRmsNormBack(a, b, eps);
}

std::unique_ptr<core::Node> CreateGroupNorm(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int nGroups = get_param_i32(tensor, 0);
    float eps = get_param_f32(tensor, 1);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateGroupNorm(a, nGroups, eps, inplace);
}

std::unique_ptr<core::Node> CreateL2Norm(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    float eps = get_param_f32(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateL2Norm(a, eps, inplace);
}

std::unique_ptr<core::Node> CreateMulMat(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    core::Prec prec = map_prec(ggml_prec(get_param_i32(tensor, 0)));
    return context->CreateMulMat(a, b, prec);
}

std::unique_ptr<core::Node> CreateMulMatId(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    core::Node *ids = get_input(tensor, 2);
    return context->CreateMulMatId(a, b, ids);
}

std::unique_ptr<core::Node> CreateOutProd(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    return context->CreateOutProd(a, b);
}

std::unique_ptr<core::Node> CreateScale(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    float scale = get_param_f32(tensor, 0);
    float bias = get_param_f32(tensor, 1);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateScale(a, scale, bias, inplace);
}

std::unique_ptr<core::Node> CreateSet(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    size_t nb1 = size_t(get_param_i32(tensor, 0));
    size_t nb2 = size_t(get_param_i32(tensor, 1));
    size_t nb3 = size_t(get_param_i32(tensor, 2));
    core::Dims stride = get_stride(tensor->type, nb1, nb2, nb3);
    int offset = get_offset(tensor, 3);
    bool inplace = bool(get_param_i32(tensor, 4));
    return context->CreateSet(a, b, stride, offset, inplace);
}

std::unique_ptr<core::Node> CreateCpy(core::Context *context, ggml_tensor *tensor) {
    if (tensor->src[1] == tensor) {
        // special case: ggml_cast
        // result will own new buffer with given 'type' and shape of 'a'
        core::Node *a = get_input(tensor, 0);
        core::DataType type = map_data_type(tensor->type);
        return context->CreateCast(a, type);
    } else {
        // regular case: ggml_cpy
        // result will reuse type and buffer of 'b'
        core::Node *a = get_input(tensor, 0);
        core::Node *b = get_input(tensor, 1);
        return context->CreateCpy(a, b);
    }
}

std::unique_ptr<core::Node> CreateCont(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Dims shape = get_shape(tensor);
    return context->CreateCont(a, shape);
}

std::unique_ptr<core::Node> CreateReshape(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Dims shape = get_shape(tensor);
    return context->CreateReshape(a, shape);
}

std::unique_ptr<core::Node> CreateView(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Dims shape = get_shape(tensor);
    if (view_is_reshape(tensor)) {
        return context->CreateReshape(a, shape);
    }
    core::Dims stride = get_stride(tensor->type, tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    // ACHTUNG: Cannot use get_offset here because, by construction, offset is stored as
    //     size_t rather than int32_t in op_params buffer (see ggml/src/ggml.c)
    size_t param0 = 0;
    memcpy(&param0, tensor->op_params, sizeof(param0));
    int offset = scale_offset(tensor->type, param0);
    return context->CreateView(a, shape, stride, offset);
}

std::unique_ptr<core::Node> CreatePermute(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int axis0 = get_param_i32(tensor, 0);
    int axis1 = get_param_i32(tensor, 1);
    int axis2 = get_param_i32(tensor, 2);
    int axis3 = get_param_i32(tensor, 3);
    core::Dims axes{axis0, axis1, axis2, axis3};
    return context->CreatePermute(a, axes);
}

std::unique_ptr<core::Node> CreateTranspose(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    return context->CreateTranspose(a);
}

std::unique_ptr<core::Node> CreateGetRows(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    return context->CreateGetRows(a, b);
}

std::unique_ptr<core::Node> CreateGetRowsBack(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    int dim0 = int(tensor->ne[0]);
    int dim1 = int(tensor->ne[1]);
    core::Dims shape{dim0, dim1, 1, 1};
    return context->CreateGetRowsBack(a, b, shape);
}

std::unique_ptr<core::Node> CreateSetRows(core::Context *context, ggml_tensor *tensor) {
    // ACHTUNG: Order of inputs is weird by construction (see ggml/src/ggml.c)
    core::Node *a = get_input(tensor, 2);
    core::Node *b = get_input(tensor, 0);
    core::Node *c = get_input(tensor, 1);
    return context->CreateSetRows(a, b, c);
}

std::unique_ptr<core::Node> CreateDiag(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    return context->CreateDiag(a);
}

std::unique_ptr<core::Node> CreateDiagMaskInf(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int nPast = get_param_i32(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateDiagMaskInf(a, nPast, inplace);
}

std::unique_ptr<core::Node> CreateDiagMaskZero(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int nPast = get_param_i32(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateDiagMaskZero(a, nPast, inplace);
}

std::unique_ptr<core::Node> CreateSoftMax(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *mask = get_input_opt(tensor, 1);
    core::Node *sinks = get_input_opt(tensor, 2);
    float scale = get_param_f32(tensor, 0);
    float maxBias = get_param_f32(tensor, 1);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateSoftMax(a, mask, sinks, scale, maxBias, inplace);
}

std::unique_ptr<core::Node> CreateSoftMaxBack(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    float scale = get_param_f32(tensor, 0);
    float maxBias = get_param_f32(tensor, 1);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateSoftMaxBack(a, b, scale, maxBias, inplace);
}

std::unique_ptr<core::Node> CreateRope(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    core::Node *c = get_input_opt(tensor, 2);
    // skip param[0] (nPast)
    int nDims = get_param_i32(tensor, 1);
    core::RopeMode mode = map_rope_mode(get_param_i32(tensor, 2));
    // skip param[3] (nCtx)
    int nCtxOrig = get_param_i32(tensor, 4);
    float freqBase = get_param_f32(tensor, 5);
    float freqScale = get_param_f32(tensor, 6);
    float extFactor = get_param_f32(tensor, 7);
    float attnFactor = get_param_f32(tensor, 8);
    float betaFast = get_param_f32(tensor, 9);
    float betaSlow = get_param_f32(tensor, 10);
    std::array<int, core::MropeSections> sections = get_mrope_sections(tensor, 11);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateRope(
        a,
        b,
        c,
        nDims,
        mode,
        nCtxOrig,
        freqBase,
        freqScale,
        extFactor,
        attnFactor,
        betaFast,
        betaSlow,
        sections, 
        inplace);
}

std::unique_ptr<core::Node> CreateRopeBack(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    core::Node *c = get_input(tensor, 2);
    // skip param[0] (nPast)
    int nDims = get_param_i32(tensor, 1);
    core::RopeMode mode = map_rope_mode(get_param_i32(tensor, 2));
    // skip param[3] (nCtx)
    int nCtxOrig = get_param_i32(tensor, 4);
    float freqBase = get_param_f32(tensor, 5);
    float freqScale = get_param_f32(tensor, 6);
    float extFactor = get_param_f32(tensor, 7);
    float attnFactor = get_param_f32(tensor, 8);
    float betaFast = get_param_f32(tensor, 9);
    float betaSlow = get_param_f32(tensor, 10);
    std::array<int, core::MropeSections> sections = get_mrope_sections(tensor, 11);
    return context->CreateRopeBack(
        a,
        b,
        c,
        nDims,
        mode,
        nCtxOrig,
        freqBase,
        freqScale,
        extFactor,
        attnFactor,
        betaFast,
        betaSlow,
        sections);
}

std::unique_ptr<core::Node> CreateClamp(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    float min = get_param_f32(tensor, 0);
    float max = get_param_f32(tensor, 1);
    return context->CreateClamp(a, min, max);
}

std::unique_ptr<core::Node> CreateConvTranspose1d(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    int s0 = get_param_i32(tensor, 0);
    int p0 = get_param_i32(tensor, 1);    // reserved: currently p0 = 0
    int d0 = get_param_i32(tensor, 2);    // reserved: currently d0 = 1
    return context->CreateConvTranspose1d(a, b, s0, p0, d0);
}

std::unique_ptr<core::Node> CreateIm2col(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    int s0 = get_param_i32(tensor, 0);
    int s1 = get_param_i32(tensor, 1);
    int p0 = get_param_i32(tensor, 2);
    int p1 = get_param_i32(tensor, 3);
    int d0 = get_param_i32(tensor, 4);
    int d1 = get_param_i32(tensor, 5);
    bool is2d = bool(get_param_i32(tensor, 6));
    core::DataType dstType = map_data_type(tensor->type);
    return context->CreateIm2col(a, b, s0, s1, p0, p1, d0, d1, is2d, dstType);
}

std::unique_ptr<core::Node> CreateIm2colBack(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    core::Dims shape = get_shape(tensor);
    int s0 = get_param_i32(tensor, 0);
    int s1 = get_param_i32(tensor, 1);
    int p0 = get_param_i32(tensor, 2);
    int p1 = get_param_i32(tensor, 3);
    int d0 = get_param_i32(tensor, 4);
    int d1 = get_param_i32(tensor, 5);
    bool is2d = bool(get_param_i32(tensor, 6));
    // result type is fixed to F32 
    return context->CreateIm2colBack(a, b, shape, s0, s1, p0, p1, d0, d1, is2d);
}

std::unique_ptr<core::Node> CreateIm2col3d(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    int s0 = get_param_i32(tensor, 0);
    int s1 = get_param_i32(tensor, 1);
    int s2 = get_param_i32(tensor, 2);
    int p0 = get_param_i32(tensor, 3);
    int p1 = get_param_i32(tensor, 4);
    int p2 = get_param_i32(tensor, 5);
    int d0 = get_param_i32(tensor, 6);
    int d1 = get_param_i32(tensor, 7);
    int d2 = get_param_i32(tensor, 8);
    int IC = get_param_i32(tensor, 9);
    core::DataType dstType = map_data_type(tensor->type);
    return context->CreateIm2col3d(a, b, s0, s1, s2, p0, p1, p2, d0, d1, d2, IC, dstType);
}

std::unique_ptr<core::Node> CreateConv2d(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    int s0 = get_param_i32(tensor, 0);
    int s1 = get_param_i32(tensor, 1);
    int p0 = get_param_i32(tensor, 2);
    int p1 = get_param_i32(tensor, 3);
    int d0 = get_param_i32(tensor, 4);
    int d1 = get_param_i32(tensor, 5);
    return context->CreateConv2d(a, b, s0, s1, p0, p1, d0, d1);
}

std::unique_ptr<core::Node> CreateConv3d(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    int s0 = get_param_i32(tensor, 0);
    int s1 = get_param_i32(tensor, 1);
    int s2 = get_param_i32(tensor, 2);
    int p0 = get_param_i32(tensor, 3);
    int p1 = get_param_i32(tensor, 4);
    int p2 = get_param_i32(tensor, 5);
    int d0 = get_param_i32(tensor, 6);
    int d1 = get_param_i32(tensor, 7);
    int d2 = get_param_i32(tensor, 8);
    int C = get_param_i32(tensor, 9);
    int N = get_param_i32(tensor, 10);
    int OC = get_param_i32(tensor, 11);
    return context->CreateConv3d(a, b, s0, s1, s2, p0, p1, p2, d0, d1, d2, C, N, OC);
}

std::unique_ptr<core::Node> CreateConv2dDw(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    int s0 = get_param_i32(tensor, 0);
    int s1 = get_param_i32(tensor, 1);
    int p0 = get_param_i32(tensor, 2);
    int p1 = get_param_i32(tensor, 3);
    int d0 = get_param_i32(tensor, 4);
    int d1 = get_param_i32(tensor, 5);
    return context->CreateConv2dDw(a, b, s0, s1, p0, p1, d0, d1);
}

std::unique_ptr<core::Node> CreateConvTranspose2d(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    // ACHTUNG: Limited support: currently s0 = s1 = stride, p0 = p1 = 0, d0 = d1 = 0
    int stride = get_param_i32(tensor, 0);
    return context->CreateConvTranspose2d(a, b, stride);
}

std::unique_ptr<core::Node> CreatePool1d(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::PoolOp op = map_pool_op(ggml_op_pool(get_param_i32(tensor, 0)));
    int k0 = get_param_i32(tensor, 1);
    int s0 = get_param_i32(tensor, 2);
    int p0 = get_param_i32(tensor, 3);
    return context->CreatePool1d(a, op, k0, s0, p0);
}

std::unique_ptr<core::Node> CreatePool2d(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::PoolOp op = map_pool_op(ggml_op_pool(get_param_i32(tensor, 0)));
    int k0 = get_param_i32(tensor, 1);
    int k1 = get_param_i32(tensor, 2);
    int s0 = get_param_i32(tensor, 3);
    int s1 = get_param_i32(tensor, 4);
    int p0 = get_param_i32(tensor, 5);
    int p1 = get_param_i32(tensor, 6);
    return context->CreatePool2d(a, op, k0, k1, s0, s1, p0, p1);
}

std::unique_ptr<core::Node> CreatePool2dBack(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *af = get_input(tensor, 1);
    core::PoolOp op = map_pool_op(ggml_op_pool(get_param_i32(tensor, 0)));
    int k0 = get_param_i32(tensor, 1);
    int k1 = get_param_i32(tensor, 2);
    int s0 = get_param_i32(tensor, 3);
    int s1 = get_param_i32(tensor, 4);
    int p0 = get_param_i32(tensor, 5);
    int p1 = get_param_i32(tensor, 6);
    return context->CreatePool2dBack(a, af, op, k0, k1, s0, s1, p0, p1);
}

std::unique_ptr<core::Node> CreateUpscale(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Dims shape = get_shape(tensor);
    int param0 = get_param_i32(tensor, 0);
    int param0_mode = param0 & ~GGML_SCALE_FLAG_ALIGN_CORNERS;
    int param0_align_corners = param0 & GGML_SCALE_FLAG_ALIGN_CORNERS;
    core::ScaleMode mode = map_scale_mode(ggml_scale_mode(param0_mode));
    bool alignCorners = (param0_align_corners != 0);
    return context->CreateUpscale(a, shape, mode, alignCorners);
}

std::unique_ptr<core::Node> CreatePad(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int lp0 = get_param_i32(tensor, 0);
    int rp0 = get_param_i32(tensor, 1);
    int lp1 = get_param_i32(tensor, 2);
    int rp1 = get_param_i32(tensor, 3);
    int lp2 = get_param_i32(tensor, 4);
    int rp2 = get_param_i32(tensor, 5);
    int lp3 = get_param_i32(tensor, 6);
    int rp3 = get_param_i32(tensor, 7);
    bool circular = bool(get_param_i32(tensor, 8));
    return context->CreatePad(a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3, circular);
}

std::unique_ptr<core::Node> CreatePadReflect1d(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int p0 = get_param_i32(tensor, 0);
    int p1 = get_param_i32(tensor, 1);
    return context->CreatePadReflect1d(a, p0, p1);
}

std::unique_ptr<core::Node> CreateRoll(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int shift0 = get_param_i32(tensor, 0);
    int shift1 = get_param_i32(tensor, 1);
    int shift2 = get_param_i32(tensor, 2);
    int shift3 = get_param_i32(tensor, 3);
    core::Dims shift{shift0, shift1, shift2, shift3};
    return context->CreateRoll(a, shift);
}

std::unique_ptr<core::Node> CreateArange(core::Context *context, ggml_tensor *tensor) {
    float start = get_param_f32(tensor, 0);
    float stop = get_param_f32(tensor, 1);
    float step = get_param_f32(tensor, 2);
    return context->CreateArange(start, stop, step);
}

std::unique_ptr<core::Node> CreateTimestepEmbedding(core::Context *context, ggml_tensor *tensor) {
    core::Node *timesteps = get_input(tensor, 0);
    int dim = get_param_i32(tensor, 0);
    int maxPeriod = get_param_i32(tensor, 1);
    return context->CreateTimestepEmbedding(timesteps, dim, maxPeriod);
}

std::unique_ptr<core::Node> CreateArgsort(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::SortOrder order = map_sort_order(ggml_sort_order(get_param_i32(tensor, 0)));
    return context->CreateArgsort(a, order);
}

std::unique_ptr<core::Node> CreateLeakyRelu(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    float negativeSlope = get_param_f32(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateLeakyRelu(a, negativeSlope, inplace);
}

std::unique_ptr<core::Node> CreateTri(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int mode = get_param_i32(tensor, 0);
    return context->CreateTri(a, mode);
}

std::unique_ptr<core::Node> CreateFill(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    float value = get_param_f32(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateFill(a, value, inplace);
}

std::unique_ptr<core::Node> CreateFlashAttnExt(core::Context *context, ggml_tensor *tensor) {
    core::Node *q = get_input(tensor, 0);
    core::Node *k = get_input(tensor, 1);
    core::Node *v = get_input(tensor, 2);
    core::Node *mask = get_input_opt(tensor, 3);
    core::Node *sinks = get_input_opt(tensor, 4);
    float scale = get_param_f32(tensor, 0);
    float maxBias = get_param_f32(tensor, 1);
    float logitSoftcap = get_param_f32(tensor, 2);
    core::Prec prec = map_prec(ggml_prec(get_param_f32(tensor, 3)));
    return context->CreateFlashAttnExt(
        q, 
        k, 
        v, 
        mask, 
        sinks, 
        scale, 
        maxBias, 
        logitSoftcap, 
        prec);
}

std::unique_ptr<core::Node> CreateFlashAttnBack(core::Context *context, ggml_tensor *tensor) {
    core::Node *q = get_input(tensor, 0);
    core::Node *k = get_input(tensor, 1);
    core::Node *v = get_input(tensor, 2);
    core::Node *d = get_input(tensor, 3);
    bool masked = bool(get_param_i32(tensor, 0));
    return context->CreateFlashAttnBack(q, k, v, d, masked);
}

std::unique_ptr<core::Node> CreateSsmConv(core::Context *context, ggml_tensor *tensor) {
    core::Node *sx = get_input(tensor, 0);
    core::Node *c = get_input(tensor, 1);
    return context->CreateSsmConv(sx, c);
}

std::unique_ptr<core::Node> CreateSsmScan(core::Context *context, ggml_tensor *tensor) {
    core::Node *s = get_input(tensor, 0);
    core::Node *x = get_input(tensor, 1);
    core::Node *dt = get_input(tensor, 2);
    core::Node *A = get_input(tensor, 3);
    core::Node *B = get_input(tensor, 4);
    core::Node *C = get_input(tensor, 5);
    core::Node *ids = get_input(tensor, 6);
    return context->CreateSsmScan(s, x, dt, A, B, C, ids);
}

std::unique_ptr<core::Node> CreateWinPart(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int np0 = get_param_i32(tensor, 0);
    int np1 = get_param_i32(tensor, 1);
    int w = get_param_i32(tensor, 2);
    return context->CreateWinPart(a, np0, np1, w);
}

std::unique_ptr<core::Node> CreateWinUnpart(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int w0 = int(tensor->ne[1]);
    int h0 = int(tensor->ne[2]);
    int w = get_param_i32(tensor, 0);
    return context->CreateWinUnpart(a, w0, h0, w);
}

std::unique_ptr<core::Node> CreateGetRelPos(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    int qh = int(tensor->ne[2]);
    int kh = int(tensor->ne[1]);
    return context->CreateGetRelPos(a, qh, kh);
}

std::unique_ptr<core::Node> CreateAddRelPos(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *pw = get_input(tensor, 1);
    core::Node *ph = get_input(tensor, 2);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateAddRelPos(a, pw, ph, inplace);
}

std::unique_ptr<core::Node> CreateRwkvWkv6(core::Context *context, ggml_tensor *tensor) {
    core::Node *k = get_input(tensor, 0);
    core::Node *v = get_input(tensor, 1);
    core::Node *r = get_input(tensor, 2);
    core::Node *tf = get_input(tensor, 3);
    core::Node *td = get_input(tensor, 4);
    core::Node *state = get_input(tensor, 5);
    return context->CreateRwkvWkv6(k, v, r, tf, td, state);
}

std::unique_ptr<core::Node> CreateGatedLinearAttn(core::Context *context, ggml_tensor *tensor) {
    core::Node *k = get_input(tensor, 0);
    core::Node *v = get_input(tensor, 1);
    core::Node *q = get_input(tensor, 2);
    core::Node *g = get_input(tensor, 3);
    core::Node *state = get_input(tensor, 4);
    float scale = get_param_f32(tensor, 0);
    return context->CreateGatedLinearAttn(k, v, q, g, state, scale);
}

std::unique_ptr<core::Node> CreateRwkvWkv7(core::Context *context, ggml_tensor *tensor) {
    core::Node *r = get_input(tensor, 0);
    core::Node *w = get_input(tensor, 1);
    core::Node *k = get_input(tensor, 2);
    core::Node *v = get_input(tensor, 3);
    core::Node *a = get_input(tensor, 4);
    core::Node *b = get_input(tensor, 5);
    core::Node *state = get_input(tensor, 6);
    return context->CreateRwkvWkv7(r, w, k, v, a, b, state);
}

std::unique_ptr<core::Node> CreateSolveTri(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    // GGML currently supports only this combination
    bool lower = true;
    bool left = true;
    bool uni = false;
    return context->CreateSolveTri(a, b, left, lower, uni);
}

std::unique_ptr<core::Node> CreateGatedDeltaNet(core::Context *context, ggml_tensor *tensor) {
    core::Node *q = get_input(tensor, 0);
    core::Node *k = get_input(tensor, 1);
    core::Node *v = get_input(tensor, 2);
    core::Node *g = get_input(tensor, 3);
    core::Node *beta = get_input(tensor, 4);
    core::Node *state = get_input(tensor, 5);
    return context->CreateGatedDeltaNet(q, k, v, g, beta, state);
}

std::unique_ptr<core::Node> CreateCrossEntropyLoss(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    return context->CreateCrossEntropyLoss(a, b);
}

std::unique_ptr<core::Node> CreateCrossEntropyLossBack(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input(tensor, 1);
    core::Node *c = get_input(tensor, 2);
    return context->CreateCrossEntropyLossBack(a, b, c);
}

std::unique_ptr<core::Node> CreateOptStepAdamw(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *grad = get_input(tensor, 1);
    core::Node *m = get_input(tensor, 2);
    core::Node *v = get_input(tensor, 3);
    core::Node *params = get_input(tensor, 4);
    return context->CreateOptStepAdamw(a, grad, m, v, params);
}

std::unique_ptr<core::Node> CreateOptStepSgd(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *grad = get_input(tensor, 1);
    core::Node *params = get_input(tensor, 2);
    return context->CreateOptStepSgd(a, grad, params);
}

// unary op node factories

std::unique_ptr<core::Node> CreateAbs(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateAbs(a, inplace);
}

std::unique_ptr<core::Node> CreateSgn(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateSgn(a, inplace);
}

std::unique_ptr<core::Node> CreateNeg(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateNeg(a, inplace);
}

std::unique_ptr<core::Node> CreateStep(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateStep(a, inplace);
}

std::unique_ptr<core::Node> CreateTanh(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateTanh(a, inplace);
}

std::unique_ptr<core::Node> CreateElu(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateElu(a, inplace);
}

std::unique_ptr<core::Node> CreateRelu(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateRelu(a, inplace);
}

std::unique_ptr<core::Node> CreateSigmoid(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateSigmoid(a, inplace);
}

std::unique_ptr<core::Node> CreateGelu(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateGelu(a, inplace);
}

std::unique_ptr<core::Node> CreateGeluQuick(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateGeluQuick(a, inplace);
}

std::unique_ptr<core::Node> CreateSilu(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateSilu(a, inplace);
}

std::unique_ptr<core::Node> CreateHardswish(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    return context->CreateHardswish(a);
}

std::unique_ptr<core::Node> CreateHardsigmoid(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    return context->CreateHardsigmoid(a);
}

std::unique_ptr<core::Node> CreateExp(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateExp(a, inplace);
}

std::unique_ptr<core::Node> CreateExpm1(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateExpm1(a, inplace);
}

std::unique_ptr<core::Node> CreateSoftplus(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateSoftplus(a, inplace);
}

std::unique_ptr<core::Node> CreateGeluErf(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateGeluErf(a, inplace);
}

std::unique_ptr<core::Node> CreateXielu(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    float alphaN = get_param_f32(tensor, 1);
    float alphaP = get_param_f32(tensor, 2);
    float beta = get_param_f32(tensor, 3);
    float eps = get_param_f32(tensor, 4);
    return context->CreateXielu(a, alphaN, alphaP, beta, eps);
}

std::unique_ptr<core::Node> CreateFloor(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateFloor(a, inplace);
}

std::unique_ptr<core::Node> CreateCeil(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateCeil(a, inplace);
}

std::unique_ptr<core::Node> CreateRound(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateRound(a, inplace);
}

std::unique_ptr<core::Node> CreateTrunc(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    bool inplace = is_inplace(tensor, 0);
    return context->CreateTrunc(a, inplace);
}

// GLU op node factories

std::unique_ptr<core::Node> CreateReglu(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input_opt(tensor, 1);
    bool swapped = bool(get_param_i32(tensor, 1));
    return context->CreateReglu(a, b, swapped);
}

std::unique_ptr<core::Node> CreateGeglu(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input_opt(tensor, 1);
    bool swapped = bool(get_param_i32(tensor, 1));
    return context->CreateGeglu(a, b, swapped);
}

std::unique_ptr<core::Node> CreateSwiglu(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input_opt(tensor, 1);
    bool swapped = bool(get_param_i32(tensor, 1));
    return context->CreateSwiglu(a, b, swapped);
}

std::unique_ptr<core::Node> CreateSwigluOai(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input_opt(tensor, 1);
    bool swapped = bool(get_param_i32(tensor, 1));
    float alpha = get_param_f32(tensor, 2);
    float limit = get_param_f32(tensor, 3);
    return context->CreateSwigluOai(a, b, swapped, alpha, limit);
}

std::unique_ptr<core::Node> CreateGegluErf(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input_opt(tensor, 1);
    bool swapped = bool(get_param_i32(tensor, 1));
    return context->CreateGegluErf(a, b, swapped);
}

std::unique_ptr<core::Node> CreateGegluQuick(core::Context *context, ggml_tensor *tensor) {
    core::Node *a = get_input(tensor, 0);
    core::Node *b = get_input_opt(tensor, 1);
    bool swapped = bool(get_param_i32(tensor, 1));
    return context->CreateGegluQuick(a, b, swapped);
}

//
//    Operation -> node factory maps
//

using NodeFactory = std::unique_ptr<core::Node> (*)(core::Context *, ggml_tensor *);

bool g_op_map_valid = false;
NodeFactory g_op_map[GGML_OP_COUNT];
NodeFactory g_unary_op_map[GGML_UNARY_OP_COUNT];
NodeFactory g_glu_op_map[GGML_GLU_OP_COUNT];

void init_op_map() {
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        g_op_map[i] = nullptr;
    }
    g_op_map[GGML_OP_NONE] = CreateTensor;
    g_op_map[GGML_OP_DUP] = CreateDup;
    g_op_map[GGML_OP_ADD] = CreateAdd;
    g_op_map[GGML_OP_ADD_ID] = CreateAddId;
    g_op_map[GGML_OP_ADD1] = CreateAdd1;
    g_op_map[GGML_OP_ACC] = CreateAcc;
    g_op_map[GGML_OP_SUB] = CreateSub;
    g_op_map[GGML_OP_MUL] = CreateMul;
    g_op_map[GGML_OP_DIV] = CreateDiv;
    g_op_map[GGML_OP_SQR] = CreateSqr;
    g_op_map[GGML_OP_SQRT] = CreateSqrt;
    g_op_map[GGML_OP_LOG] = CreateLog;
    g_op_map[GGML_OP_SIN] = CreateSin;
    g_op_map[GGML_OP_COS] = CreateCos;
    g_op_map[GGML_OP_SUM] = CreateSum;
    g_op_map[GGML_OP_SUM_ROWS] = CreateSumRows;
    g_op_map[GGML_OP_CUMSUM] = CreateCumSum;
    g_op_map[GGML_OP_MEAN] = CreateMean;
    g_op_map[GGML_OP_ARGMAX] = CreateArgmax;
    g_op_map[GGML_OP_COUNT_EQUAL] = CreateCountEqual;
    g_op_map[GGML_OP_REPEAT] = CreateRepeat;
    g_op_map[GGML_OP_REPEAT_BACK] = CreateRepeatBack;
    g_op_map[GGML_OP_CONCAT] = CreateConcat;
    g_op_map[GGML_OP_SILU_BACK] = CreateSiluBack;
    g_op_map[GGML_OP_NORM] = CreateNorm;
    g_op_map[GGML_OP_RMS_NORM] = CreateRmsNorm;
    g_op_map[GGML_OP_RMS_NORM_BACK] = CreateRmsNormBack;
    g_op_map[GGML_OP_GROUP_NORM] = CreateGroupNorm;
    g_op_map[GGML_OP_L2_NORM] = CreateL2Norm;
    g_op_map[GGML_OP_MUL_MAT] = CreateMulMat;
    g_op_map[GGML_OP_MUL_MAT_ID] = CreateMulMatId;
    g_op_map[GGML_OP_OUT_PROD] = CreateOutProd;
    g_op_map[GGML_OP_SCALE] = CreateScale;
    g_op_map[GGML_OP_SET] = CreateSet;
    g_op_map[GGML_OP_CPY] = CreateCpy;
    g_op_map[GGML_OP_CONT] = CreateCont;
    g_op_map[GGML_OP_RESHAPE] = CreateReshape;
    g_op_map[GGML_OP_VIEW] = CreateView;
    g_op_map[GGML_OP_PERMUTE] = CreatePermute;
    g_op_map[GGML_OP_TRANSPOSE] = CreateTranspose;
    g_op_map[GGML_OP_GET_ROWS] = CreateGetRows;
    g_op_map[GGML_OP_GET_ROWS_BACK] = CreateGetRowsBack;
    g_op_map[GGML_OP_SET_ROWS] = CreateSetRows;
    g_op_map[GGML_OP_DIAG] = CreateDiag;
    g_op_map[GGML_OP_DIAG_MASK_INF] = CreateDiagMaskInf;
    g_op_map[GGML_OP_DIAG_MASK_ZERO] = CreateDiagMaskZero;
    g_op_map[GGML_OP_SOFT_MAX] = CreateSoftMax;
    g_op_map[GGML_OP_SOFT_MAX_BACK] = CreateSoftMaxBack;
    g_op_map[GGML_OP_ROPE] = CreateRope;
    g_op_map[GGML_OP_ROPE_BACK] = CreateRopeBack;
    g_op_map[GGML_OP_CLAMP] = CreateClamp;
    g_op_map[GGML_OP_CONV_TRANSPOSE_1D] = CreateConvTranspose1d;
    g_op_map[GGML_OP_IM2COL] = CreateIm2col;
    g_op_map[GGML_OP_IM2COL_BACK] = CreateIm2colBack;
    g_op_map[GGML_OP_IM2COL_3D] = CreateIm2col3d;
    g_op_map[GGML_OP_CONV_2D] = CreateConv2d;
    g_op_map[GGML_OP_CONV_3D] = CreateConv3d;
    g_op_map[GGML_OP_CONV_2D_DW] = CreateConv2dDw;
    g_op_map[GGML_OP_CONV_TRANSPOSE_2D] = CreateConvTranspose2d;
    g_op_map[GGML_OP_POOL_1D] = CreatePool1d;
    g_op_map[GGML_OP_POOL_2D] = CreatePool2d;
    g_op_map[GGML_OP_POOL_2D_BACK] = CreatePool2dBack;
    g_op_map[GGML_OP_UPSCALE] = CreateUpscale;
    g_op_map[GGML_OP_PAD] = CreatePad;
    g_op_map[GGML_OP_PAD_REFLECT_1D] = CreatePadReflect1d;
    g_op_map[GGML_OP_ROLL] = CreateRoll;
    g_op_map[GGML_OP_ARANGE] = CreateArange;
    g_op_map[GGML_OP_TIMESTEP_EMBEDDING] = CreateTimestepEmbedding;
    g_op_map[GGML_OP_ARGSORT] = CreateArgsort;
    g_op_map[GGML_OP_LEAKY_RELU] = CreateLeakyRelu;
    g_op_map[GGML_OP_TRI] = CreateTri;
    g_op_map[GGML_OP_FILL] = CreateFill;
    g_op_map[GGML_OP_FLASH_ATTN_EXT] = CreateFlashAttnExt;
    g_op_map[GGML_OP_FLASH_ATTN_BACK] = CreateFlashAttnBack;
    g_op_map[GGML_OP_SSM_CONV] = CreateSsmConv;
    g_op_map[GGML_OP_SSM_SCAN] = CreateSsmScan;
    g_op_map[GGML_OP_WIN_PART] = CreateWinPart;
    g_op_map[GGML_OP_WIN_UNPART] = CreateWinUnpart;
    g_op_map[GGML_OP_GET_REL_POS] = CreateGetRelPos;
    g_op_map[GGML_OP_ADD_REL_POS] = CreateAddRelPos;
    g_op_map[GGML_OP_RWKV_WKV6] = CreateRwkvWkv6;
    g_op_map[GGML_OP_GATED_LINEAR_ATTN] = CreateGatedLinearAttn;
    g_op_map[GGML_OP_RWKV_WKV7] = CreateRwkvWkv7;
    g_op_map[GGML_OP_SOLVE_TRI] = CreateSolveTri;
#ifdef ENABLE_GATED_DELTA_NET
    g_op_map[GGML_OP_GATED_DELTA_NET] = CreateGatedDeltaNet;
#endif
    g_op_map[GGML_OP_UNARY] = nullptr; // use g_unary_op_map
    g_op_map[GGML_OP_MAP_CUSTOM1] = nullptr;
    g_op_map[GGML_OP_MAP_CUSTOM2] = nullptr;
    g_op_map[GGML_OP_MAP_CUSTOM3] = nullptr;
    g_op_map[GGML_OP_CUSTOM] = nullptr;
    g_op_map[GGML_OP_CROSS_ENTROPY_LOSS] = CreateCrossEntropyLoss;
    g_op_map[GGML_OP_CROSS_ENTROPY_LOSS_BACK] = CreateCrossEntropyLossBack;
    g_op_map[GGML_OP_OPT_STEP_ADAMW] = CreateOptStepAdamw;
    g_op_map[GGML_OP_OPT_STEP_SGD] = CreateOptStepSgd;
    g_op_map[GGML_OP_GLU] = nullptr; // use g_glu_op_map
}

void init_unary_op_map() {
    for (int i = 0; i < GGML_UNARY_OP_COUNT; i++) {
        g_unary_op_map[i] = nullptr;
    }
    g_unary_op_map[GGML_UNARY_OP_ABS] = CreateAbs;
    g_unary_op_map[GGML_UNARY_OP_SGN] = CreateSgn;
    g_unary_op_map[GGML_UNARY_OP_NEG] = CreateNeg;
    g_unary_op_map[GGML_UNARY_OP_STEP] = CreateStep;
    g_unary_op_map[GGML_UNARY_OP_TANH] = CreateTanh;
    g_unary_op_map[GGML_UNARY_OP_ELU] = CreateElu;
    g_unary_op_map[GGML_UNARY_OP_RELU] = CreateRelu;
    g_unary_op_map[GGML_UNARY_OP_SIGMOID] = CreateSigmoid;
    g_unary_op_map[GGML_UNARY_OP_GELU] = CreateGelu;
    g_unary_op_map[GGML_UNARY_OP_GELU_QUICK] = CreateGeluQuick;
    g_unary_op_map[GGML_UNARY_OP_SILU] = CreateSilu;
    g_unary_op_map[GGML_UNARY_OP_HARDSWISH] = CreateHardswish;
    g_unary_op_map[GGML_UNARY_OP_HARDSIGMOID] = CreateHardsigmoid;
    g_unary_op_map[GGML_UNARY_OP_EXP] = CreateExp;
    g_unary_op_map[GGML_UNARY_OP_EXPM1] = CreateExpm1;
    g_unary_op_map[GGML_UNARY_OP_SOFTPLUS] = CreateSoftplus;
    g_unary_op_map[GGML_UNARY_OP_GELU_ERF] = CreateGeluErf;
    g_unary_op_map[GGML_UNARY_OP_XIELU] = CreateXielu;
    g_unary_op_map[GGML_UNARY_OP_FLOOR] = CreateFloor;
    g_unary_op_map[GGML_UNARY_OP_CEIL] = CreateCeil;
    g_unary_op_map[GGML_UNARY_OP_ROUND] = CreateRound;
    g_unary_op_map[GGML_UNARY_OP_TRUNC] = CreateTrunc; 
}

void init_glu_op_map() {
    for (int i = 0; i < GGML_GLU_OP_COUNT; i++) {
        g_glu_op_map[i] = nullptr;
    }
    g_glu_op_map[GGML_GLU_OP_REGLU] = CreateReglu;
    g_glu_op_map[GGML_GLU_OP_GEGLU] = CreateGeglu;
    g_glu_op_map[GGML_GLU_OP_SWIGLU] = CreateSwiglu;
    g_glu_op_map[GGML_GLU_OP_SWIGLU_OAI] = CreateSwigluOai;
    g_glu_op_map[GGML_GLU_OP_GEGLU_ERF] = CreateGegluErf;
    g_glu_op_map[GGML_GLU_OP_GEGLU_QUICK] = CreateGegluQuick; 
}

NodeFactory get_node_factory(const ggml_tensor *tensor) {
    int op = tensor->op;
    assert(op >= 0 && op < GGML_OP_COUNT);
    if (op == GGML_OP_UNARY) {
        int unary_op = ggml_get_unary_op(tensor);
        assert(unary_op >= 0 && unary_op < GGML_UNARY_OP_COUNT);
        return g_unary_op_map[unary_op];
    }
    if (op == GGML_OP_GLU) {
        int glu_op = ggml_get_glu_op(tensor);
        assert(glu_op >= 0 && glu_op < GGML_GLU_OP_COUNT);
        return g_glu_op_map[glu_op];
    }
    return g_op_map[op];
}

//
//    Common backend support validation
//

bool validate_mul_mat_bcast(const ggml_tensor *src0, const ggml_tensor *src1) {
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        int64_t dim0 = src0->ne[i];
        int64_t dim1 = src1->ne[i];
        if (dim0 != dim1 && dim1 != 1) {
            return false;
        }
    }
    return true;
}

bool validate_type(const ggml_tensor *tensor, const std::vector<ggml_type> &types) {
    if (tensor == nullptr) {
        return true;
    }
    for (ggml_type type: types) {
        if (tensor->type == type) {
            return true;
        }
    }
    return false;
}

bool validate_tensor_type(const ggml_tensor *tensor, const std::vector<ggml_type> &types) {
    return (validate_type(tensor, types) && 
        validate_type(tensor->src[0], types) &&
        validate_type(tensor->src[1], types));
}

bool skip_tensor_type_validation(const ggml_tensor *tensor) {
    switch (tensor->op) {
    case GGML_OP_NONE:
    case GGML_OP_CONCAT:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
        return true;
    default:
        return false;
    }
}

std::vector<ggml_type> g_types_f32{GGML_TYPE_F32};

std::vector<ggml_type> g_types_float{GGML_TYPE_F16, GGML_TYPE_F32};

std::vector<ggml_type> g_types_quant_v0{
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_MXFP4
};

std::vector<ggml_type> g_types_quant{
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_MXFP4
};

bool supports_data_type_common(ggml_backend_dev_t dev, const ggml_tensor *tensor) {
    // ACHTUNG: This is temporary crude solution based on some conservative assumptions
    //     about data types commonly supported by Intel CPUs and GPUs
    enum ggml_backend_dev_type dev_type = dev->iface.get_type(dev);
    enum ggml_type dst_type = tensor->type;
    const ggml_tensor *src0 = tensor->src[0];
    const ggml_tensor *src1 = tensor->src[1];
    const ggml_tensor *src2 = tensor->src[2];
    const ggml_tensor *src3 = tensor->src[3];
    const ggml_tensor *src4 = tensor->src[4];
    if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        if (skip_tensor_type_validation(tensor)) {
            return true;
        } else {
            return validate_tensor_type(tensor, g_types_f32);
        }
    } else if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
        if (skip_tensor_type_validation(tensor)) {
            return true;
        } 
        switch (tensor->op) {
        case GGML_OP_UNARY:
            return true;
        case GGML_OP_CUMSUM:
            if (src0->type != GGML_TYPE_F32) {
                return false;
            }
            // Temporary restriction (for compatibility with UHD)
            // TODO: Make this generic
            if (src0->ne[0] > 256 * 256) {
                return false;
            }
            return true;
        case GGML_OP_MUL_MAT:
            if (!validate_type(src0, g_types_float) &&
                    !validate_type(src0, g_types_quant)) {
                return false;
            }
            if (!validate_type(src1, g_types_float)) {
                return false;
            }
            return true;
        case GGML_OP_MUL_MAT_ID:
            if ((src0->type == GGML_TYPE_F32 ||
                        src0->type == GGML_TYPE_F16) &&
                    src1->type == GGML_TYPE_F32 &&
                    src2->type == GGML_TYPE_I32) {
                if (src1->ne[2] > 8 && src0->type != GGML_TYPE_F32) {
                    return false;
                }
                if (src0->ne[0] % 2 != 0) {
                    return false;
                }
                return true;
            }
            if (validate_type(src0, g_types_quant) &&
                    src1->type == GGML_TYPE_F32 &&
                    src2->type == GGML_TYPE_I32) {
                return true;
            }
            return false;
        case GGML_OP_SET:
            // TODO: Implement SET for all supported quant types
            if (validate_type(src0, g_types_quant_v0) ||
                    validate_type(src1, g_types_quant_v0)) {
                return false;
            }
            return true;
        case GGML_OP_GET_ROWS:
            if (!validate_type(src0, g_types_float)) {
                return false;
            }
            // oneDNN restriction: I64 not supported
            if (src1->type != GGML_TYPE_I32) {
                return false;
            }
            return true;
        case GGML_OP_SET_ROWS:
            if (!validate_type(src2, g_types_float)) {
                return false;
            }
            if (src0->type != GGML_TYPE_F32) {
                return false;
            }
            if (src1->type != GGML_TYPE_I32 &&
                    src1->type != GGML_TYPE_I64) {
                return false;
            }
            return true;
        case GGML_OP_ROPE:
            // TODO
            return true;
        case GGML_OP_ARGSORT:
            if (src0->type != GGML_TYPE_F32 || tensor->type != GGML_TYPE_I32) {
                return false;
            }
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            if (src0->type != GGML_TYPE_F32) {
                return false;
            }
            if (src1->type != GGML_TYPE_F16) {
                return false;
            }
            if (src2->type != GGML_TYPE_F16) {
                return false;
            }
            if (src3 != nullptr && src3->type != GGML_TYPE_F16) {
                return false;
            }
            if (src4 != nullptr && src4->type != GGML_TYPE_F32) {
                return false;
            }
            return true;
        default:
            return validate_tensor_type(tensor, g_types_float);
        }
    } else {
        return false;
    }
}

bool supports_op_common(ggml_backend_dev_t dev, const ggml_tensor *tensor) {
    const ggml_tensor *src0 = tensor->src[0];
    const ggml_tensor *src1 = tensor->src[1];
    const ggml_tensor *src2 = tensor->src[2];
    switch (tensor->op) {
    case GGML_OP_SUM:
        // Temporary restriction of reference GGML CPU backend
        if (src0->nb[0] != ggml_type_size(src0->type)) {
            return false;
        }
        return true;
    case GGML_OP_NORM:
    case GGML_OP_L2_NORM:
    case GGML_OP_RMS_NORM:
        if (!ggml_is_contiguous_rows(src0)) {
            return false;
        }
        return true;
    case GGML_OP_MUL_MAT:
        if (validate_type(src0, g_types_quant)) {
            return true;
        }
        if (!validate_mul_mat_bcast(src0, src1)) {
            return false;
        }
        return true;
    case GGML_OP_OUT_PROD:
        if (!validate_mul_mat_bcast(src0, src1)) {
            return false;
        }
        return true;
    case GGML_OP_SOFT_MAX:
        {
            if (src1 != nullptr || src2 != nullptr) {
                return false;
            }
            float scale = 1.0f;
            float max_bias = 0.0f;
            memcpy(&scale, tensor->op_params + 0, sizeof(float));
            memcpy(&max_bias, tensor->op_params + 1, sizeof(float)); 
            if (scale != 1.0f || max_bias != 0.0f) {
                return false;
            }
        }
        return true;
    case GGML_OP_UPSCALE:
        {
            // Restrictions of oneDNN resampling primitive
            // non-spatial dimensions must match
            if (tensor->ne[2] != src0->ne[2] || tensor->ne[3] != src0->ne[3]) {
                return false;
            }
            int param0 = int(tensor->op_params[0]);
            int param0_mode = param0 & ~GGML_SCALE_FLAG_ALIGN_CORNERS;
            int param0_align_corners = param0 & GGML_SCALE_FLAG_ALIGN_CORNERS;
            core::ScaleMode mode = map_scale_mode(ggml_scale_mode(param0_mode));
            bool alignCorners = (param0_align_corners != 0);
            if (mode != core::ScaleMode::Nearest && mode != core::ScaleMode::Bilinear) {
                return false;
            }
            if (alignCorners) {
                return false;
            }
        }
        return true;
    case GGML_OP_ARGSORT:
        if (src0->ne[0] > 1024) {
            // ACHTUNG: Temporary restriction
            return false;
        }
        return true;
    case GGML_OP_GATED_DELTA_NET:
        {
            int K = int(tensor->op_params[0]);
            if (K != 1) {
                // Not yet implemented
                return false;
            }
        }
        return true;
    }
    return true;
}

} // namespace

void ggml_arhat_init_op_map() {
    if (g_op_map_valid) {
        return;
    }
    init_op_map();
    init_unary_op_map();
    init_glu_op_map();
    g_op_map_valid = true;
}

bool ggml_arhat_supports_op(ggml_backend_dev_t dev, const ggml_tensor *tensor) {
    if (!supports_data_type_common(dev, tensor)) {
        log_unsupported_op(tensor, "data_type_common");
        return false;
    }
    if (!supports_op_common(dev, tensor)) {
        log_unsupported_op(tensor, "op_common");
        return false;
    }
    NodeFactory factory = get_node_factory(tensor);
    if (factory == nullptr) {
        log_unsupported_op(tensor, "factory");
        return false;
    }
    return true;
}

std::unique_ptr<core::Node> ggml_arhat_create_node(
        core::Context *context,
        ggml_tensor *tensor, 
        int bufferIndex,
        size_t tensorAddr) {
    size_t addr = core::Context::NULL_BUFFER_ADDR;
    if (tensor->op != GGML_OP_NONE &&
            tensor->op != GGML_OP_VIEW &&
            tensor->op != GGML_OP_TRANSPOSE &&
            tensor->op != GGML_OP_PERMUTE &&
            tensor->op != GGML_OP_RESHAPE &&
            tensor->op != GGML_OP_SET_ROWS) {
        addr = tensorAddr;
    }
    context->SetBuffer(bufferIndex, addr);
    NodeFactory factory = get_node_factory(tensor);
    GGML_ASSERT(factory != nullptr);
    return factory(context, tensor);
}

