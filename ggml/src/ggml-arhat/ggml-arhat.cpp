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

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <mutex>

#include "ggml-impl.h"
#include "ggml-backend.h" 
#include "ggml-backend-impl.h" 
#include "ggml-arhat.h"

#include "arhat/core/runtime.hpp"
#include "arhat/onednn/base/setup.hpp"
#include "arhat/onednn/gpu/setup.hpp"

#include "ggml-arhat/ops.hpp"
#include "ggml-arhat/monitor.hpp"

namespace core = arhat::core;

//
//    Consants
//

static constexpr size_t ggml_backend_arhat_dummy_alignment = 128;

//
//    Forward declarations
//

struct ggml_backend_arhat_buffer_context;
struct ggml_backend_arhat_context;

static ggml_backend_arhat_context *ggml_arhat_context_init(ggml_backend_dev_t dev);
static core::Context *ggml_backend_arhat_context_get_impl(ggml_backend_arhat_context *ctx);
static void ggml_backend_arhat_context_set_cur_buf_ctx(
    ggml_backend_arhat_context *ctx, 
    ggml_backend_arhat_buffer_context *buf_ctx);
static ggml_backend_arhat_buffer_context *
    ggml_backend_arhat_context_get_cur_buf_ctx(ggml_backend_arhat_context *ctx);
static void ggml_backend_arhat_context_reset(ggml_backend_arhat_context *ctx);

//
//    ggml_backend_arhat_buffer_context
//
//    Backend buffer
//

struct ggml_backend_arhat_buffer_context { 
    std::vector<std::unique_ptr<core::Node>> nodes;
    int bufferIndex;
};

static void ggml_backend_arhat_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_arhat_buffer_context *ctx = (ggml_backend_arhat_buffer_context *)buffer->context;
    ggml_backend_arhat_context *backend_ctx = ggml_arhat_context_init(buffer->buft->device);
    core::Context *context = ggml_backend_arhat_context_get_impl(backend_ctx);
    context->DeleteBuffer(ctx->bufferIndex);
    delete ctx;
}

static void *ggml_backend_arhat_buffer_get_base(ggml_backend_buffer_t buffer) {
    // all buffer memory allocation is performed by Arhat runtime
    // following is dummy value to deceive standard allocator in ggml-alloc.c
    return (void *)(uintptr_t)ggml_backend_arhat_dummy_alignment;
} 

static enum ggml_status ggml_backend_arhat_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor *tensor) {
    ggml_backend_arhat_context *backend_ctx = ggml_arhat_context_init(buffer->buft->device);
    ggml_backend_arhat_buffer_context *ctx = ggml_backend_arhat_context_get_cur_buf_ctx(backend_ctx);
    GGML_ASSERT(ctx != nullptr);
    core::Context *context = ggml_backend_arhat_context_get_impl(backend_ctx);
    size_t tensorAddr = (char *)tensor->data - (char *)ggml_backend_arhat_buffer_get_base(buffer);
    std::unique_ptr<core::Node> node = 
        ggml_arhat_create_node(context, tensor, ctx->bufferIndex, tensorAddr);
    core::Node *node_ptr = node.get();
    ctx->nodes.emplace_back(std::move(node));
    tensor->extra = node_ptr;
    return GGML_STATUS_SUCCESS; 
}

static void ggml_backend_arhat_buffer_set_tensor(
        ggml_backend_buffer_t buffer, 
        ggml_tensor *tensor, 
        const void *data, 
        size_t offset, 
        size_t size) { 
    core::Node *node = static_cast<core::Node *>(tensor->extra);
    node->Write(data, int(offset), int(size));
}

static void ggml_backend_arhat_buffer_get_tensor(
        ggml_backend_buffer_t buffer, 
        const ggml_tensor *tensor, 
        void *data, 
        size_t offset, 
        size_t size) { 
    core::Node *node = static_cast<core::Node *>(tensor->extra);
    node->Read(data, int(offset), int(size));
}

static void ggml_backend_arhat_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) { 
    ggml_backend_arhat_buffer_context *ctx = (ggml_backend_arhat_buffer_context *)buffer->context;
    for (auto &node: ctx->nodes) {
        node->Fill(value);
    }
}

static void ggml_backend_arhat_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_arhat_buffer_context *ctx = (ggml_backend_arhat_buffer_context *)buffer->context;
    ggml_backend_arhat_context *backend_ctx = ggml_arhat_context_init(buffer->buft->device);
    core::Context *context = ggml_backend_arhat_context_get_impl(backend_ctx);
    ggml_backend_arhat_context_set_cur_buf_ctx(backend_ctx, ctx);
    ctx->nodes.clear();
    context->ResetBuffer(ctx->bufferIndex);
    ggml_backend_arhat_context_reset(backend_ctx);
}

static ggml_backend_buffer_i ggml_backend_arhat_buffer_interface = {
    /* .free_buffer        = */ ggml_backend_arhat_buffer_free_buffer,
    /* .get_base           = */ ggml_backend_arhat_buffer_get_base,
    /* .init_tensor        = */ ggml_backend_arhat_buffer_init_tensor,
    /* .memset_tensor      = */ NULL,
    /* .set_tensor         = */ ggml_backend_arhat_buffer_set_tensor,
    /* .get_tensor         = */ ggml_backend_arhat_buffer_get_tensor,
    /* .set_tensor_2d      = */ NULL,
    /* .get_tensor_2d      = */ NULL,
    /* .cpy_tensor         = */ NULL,
    /* .clear              = */ ggml_backend_arhat_buffer_clear,
    /* .reset              = */ ggml_backend_arhat_buffer_reset,
}; 

//
//    ggml_backend_arhat_buffer_type
//
//    Backend buffer type
//

static const char *ggml_backend_arhat_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    GGML_UNUSED(buffer_type);
    return "Arhat";
} 

static ggml_backend_buffer_t ggml_backend_arhat_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buffer_type, size_t size) { 
    ggml_backend_arhat_context *backend_ctx = ggml_arhat_context_init(buffer_type->device); 
    core::Context *context = ggml_backend_arhat_context_get_impl(backend_ctx);
    ggml_backend_arhat_buffer_context *ctx = new ggml_backend_arhat_buffer_context();
    ctx->bufferIndex = context->CreateBuffer();
    ggml_backend_arhat_context_set_cur_buf_ctx(backend_ctx, ctx);
    return ggml_backend_buffer_init(buffer_type, ggml_backend_arhat_buffer_interface, ctx, size); 
}

static size_t ggml_backend_arhat_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // all buffer memory allocation is performed by Arhat runtime
    // following is arbitrarily chosen dummy value 
    GGML_UNUSED(buft);
    return ggml_backend_arhat_dummy_alignment;
} 

static ggml_backend_buffer_type_i ggml_backend_arhat_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_arhat_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_arhat_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_arhat_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX 
    /* .get_alloc_size   = */ NULL,
    /* .is_host          = */ NULL,
}; 

//
//    ggml_backend_arhat_context
//
//    Backend context
//

struct ggml_backend_arhat_context {
    std::unique_ptr<core::Context> context;
    std::unique_ptr<ArhatComputeMonitor> monitor;
    ggml_backend_arhat_buffer_context *cur_buf_ctx;
};

static core::Context *ggml_backend_arhat_context_get_impl(ggml_backend_arhat_context *ctx) {
    return ctx->context.get();
}

static void ggml_backend_arhat_context_set_cur_buf_ctx(
        ggml_backend_arhat_context *ctx, 
        ggml_backend_arhat_buffer_context *buf_ctx) {
    ctx->cur_buf_ctx = buf_ctx;
}

static ggml_backend_arhat_buffer_context *
        ggml_backend_arhat_context_get_cur_buf_ctx(ggml_backend_arhat_context *ctx) {
    return ctx->cur_buf_ctx;
}

static void ggml_backend_arhat_context_reset(ggml_backend_arhat_context *ctx) {
    ctx->context->Reset();
}

static const char *ggml_backend_arhat_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "Arhat";
}

static void ggml_backend_arhat_free(ggml_backend_t backend) {
#if 0 // TODO: Revise this
    ggml_backend_arhat_context *backend_ctx = (ggml_backend_arhat_context *)(backend->context); 
    delete backend_ctx;
    delete backend;
#endif
} 

static void ggml_backend_arhat_synchronize(ggml_backend_t backend) {
    ggml_backend_arhat_context *backend_ctx = (ggml_backend_arhat_context *)(backend->context); 
    backend_ctx->context->Wait();
}

// RESERVED
static void ggml_backend_arhat_set_tensor_async(
        ggml_backend_t backend, 
        ggml_tensor *tensor, 
        const void *data, 
        size_t offset, 
        size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

// RESERVED
static void ggml_backend_arhat_get_tensor_async(
        ggml_backend_t backend, 
        const ggml_tensor *tensor, 
        void *data, 
        size_t offset, 
        size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

// RESERVED
static bool ggml_backend_arhat_cpy_tensor_async(
        ggml_backend_t backend, 
        const ggml_tensor *src, 
        ggml_tensor *dst) {
    GGML_UNUSED(backend);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
} 

static ggml_status ggml_backend_arhat_graph_compute(ggml_backend_t backend, ggml_cgraph *cgraph) { 
    ggml_backend_arhat_context *backend_ctx = (ggml_backend_arhat_context *)(backend->context); 
    ArhatComputeMonitor *monitor = backend_ctx->monitor.get();
    monitor->StartGraph(cgraph);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i]; 
        // TODO: Figure out how to handle fusions (postponed for now)
        core::Node *arhat_node = (core::Node *)node->extra;
#if 1 // EXPERIMENTAL
        if (arhat_node == nullptr && 
//                (node->flags & GGML_TENSOR_FLAG_INPUT) != 0 &&
                (node->op == GGML_OP_VIEW || 
                    node->op == GGML_OP_RESHAPE || 
                    node->op == GGML_OP_PERMUTE || 
                    node->op == GGML_OP_TRANSPOSE)) {
            // TODO: Explain why
            continue;
        }
#endif
        GGML_ASSERT(arhat_node != nullptr);
        monitor->StartNode(i, node, arhat_node);
        arhat_node->Compute();
        monitor->EndNode();
    }
    monitor->EndGraph();
    return GGML_STATUS_SUCCESS; 
}

static const ggml_backend_i ggml_backend_arhat_interface = {
    /* .get_name                   = */ ggml_backend_arhat_name,
    /* .free                       = */ ggml_backend_arhat_free,
    /* .set_tensor_async           = */ NULL,  // ggml_backend_arhat_set_tensor_async
    /* .get_tensor_async           = */ NULL,  // ggml_backend_arhat_get_tensor_async
    /* .set_tensor_2d_async        = */ NULL,  // ggml_backend_arhat_set_tensor_2d_async
    /* .get_tensor_2d_async        = */ NULL,  // ggml_backend_arhat_get_tensor_2d_async
    /* .cpy_tensor_async           = */ NULL,  // ggml_backend_arhat_cpy_tensor_async
    /* .synchronize                = */ ggml_backend_arhat_synchronize,
    /* .graph_plan_create          = */ NULL,
    /* .graph_plan_free            = */ NULL,
    /* .graph_plan_update          = */ NULL,
    /* .graph_plan_compute         = */ NULL,
    /* .graph_compute              = */ ggml_backend_arhat_graph_compute,
    /* .event_record               = */ NULL,
    /* .event_wait                 = */ NULL,
    /* .graph_optimize             = */ NULL, 
};

static ggml_guid_t ggml_backend_arhat_guid() {
    static ggml_guid guid = { 
        0x96, 0xa7, 0xf8, 0x63, 0x79, 0x42, 0x43, 0xeb, 
        0x8f, 0x23, 0xca, 0xba, 0x28, 0xad, 0x2f, 0xfd 
    };
    return &guid;
}

//
//    ggml_backend_arhat_device_context
//
//    Backend device context
//

struct ggml_backend_arhat_device_context {
    int platform_id;
    std::string platform_name;
    int device_id;
    enum ggml_backend_dev_type device_type;
    std::string device_name;
    std::string device_description;
    ggml_backend_arhat_context *backend_ctx;
    ggml_backend_buffer_type buffer_type;
};

static ggml_backend_arhat_context *ggml_arhat_context_init(ggml_backend_dev_t dev) {
    ggml_backend_arhat_device_context *dev_ctx = (ggml_backend_arhat_device_context *)dev->context;
    if (dev_ctx->backend_ctx != nullptr) {
        return dev_ctx->backend_ctx;
    }
    ggml_backend_arhat_context *backend_ctx = new ggml_backend_arhat_context {
        /* .context     = */ nullptr,
        /* .monitor     = */ nullptr,
        /* .cur_buf_ctx = */ nullptr,
    };
    dev_ctx->backend_ctx = backend_ctx;
    core::Platform *platform = core::Platform::Get(dev_ctx->platform_id);
    core::Device *device = platform->GetDevice(dev_ctx->device_id);
    backend_ctx->context = device->CreateContext();
    backend_ctx->monitor = std::make_unique<ArhatComputeMonitor>(backend_ctx->context.get());
    return dev_ctx->backend_ctx;
}

static const char *ggml_backend_arhat_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_arhat_device_context *dev_ctx = (ggml_backend_arhat_device_context *)dev->context;
    return dev_ctx->device_name.c_str();
}

static const char *ggml_backend_arhat_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_arhat_device_context *dev_ctx = (ggml_backend_arhat_device_context *)dev->context;
    return dev_ctx->device_description.c_str();
}

static void ggml_backend_arhat_device_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total) {
    GGML_UNUSED(dev);
    // TODO: Replace this stub with actual device query
    constexpr size_t KB = size_t(1024);
    size_t dummy = size_t(24) * KB * KB * KB;
    *free = dummy;
    *total = dummy;
}

static enum ggml_backend_dev_type ggml_backend_arhat_device_get_type(ggml_backend_dev_t dev) { 
    ggml_backend_arhat_device_context *dev_ctx = (ggml_backend_arhat_device_context *)dev->context;
    return dev_ctx->device_type;
}

static void ggml_backend_arhat_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props *props) {
    props->name = ggml_backend_arhat_device_get_name(dev);
    props->description = ggml_backend_arhat_device_get_description(dev);
    props->type = ggml_backend_arhat_device_get_type(dev);
    ggml_backend_arhat_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = ggml_backend_dev_caps {
        /* .async                 = */ false,    // TODO: Check this
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
} 

static ggml_backend_t ggml_backend_arhat_device_init_backend(ggml_backend_dev_t dev, const char *params) { 
    GGML_UNUSED(params);
    ggml_backend_arhat_context *backend_ctx = ggml_arhat_context_init(dev);
    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_arhat_guid(),
        /* .interface = */ ggml_backend_arhat_interface,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    }; 
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_arhat_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_arhat_device_context *dev_ctx = (ggml_backend_arhat_device_context *)dev->context;
    // ACHTUNG: Do we really want to assign it every time?
    dev_ctx->buffer_type = ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_arhat_buffer_type_interface,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };
    return &dev_ctx->buffer_type;
} 

static ggml_backend_buffer_t ggml_backend_arhat_device_buffer_from_ptr(
        ggml_backend_dev_t dev, void *ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_arhat_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor *op) {
    return ggml_arhat_supports_op(dev, op);
}

static bool ggml_backend_arhat_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Check 'dev' and 'buffer_type' are not objects belonging to this backend
    if (dev->iface.get_name != ggml_backend_arhat_device_get_name ||
            buft->iface.get_name != ggml_backend_arhat_buffer_type_get_name) {
        return false;
    }
    // Check context is the same
    ggml_backend_arhat_context *backend_ctx0 = ggml_arhat_context_init(dev);
    ggml_backend_arhat_context *backend_ctx1 = ggml_arhat_context_init(buft->device);
    return (backend_ctx0->context == backend_ctx1->context);
} 

static struct ggml_backend_device_i ggml_backend_arhat_device_interface = {
    /* .get_name             = */ ggml_backend_arhat_device_get_name,
    /* .get_description      = */ ggml_backend_arhat_device_get_description,
    /* .get_memory           = */ ggml_backend_arhat_device_get_memory,
    /* .get_type             = */ ggml_backend_arhat_device_get_type,
    /* .get_props            = */ ggml_backend_arhat_device_get_props,
    /* .init_backend         = */ ggml_backend_arhat_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_arhat_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_arhat_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_arhat_device_supports_op,
    /* .supports_buft        = */ ggml_backend_arhat_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
}; 

//
//    ggml_backend_arhat_reg
//
//    Backend registry
//

static std::vector<ggml_backend_device> g_ggml_backend_arhat_devices; 

static enum ggml_backend_dev_type map_device_kind(core::DeviceKind device_kind) {
    switch (device_kind) {
    case core::DeviceKind::Cpu:
        return GGML_BACKEND_DEVICE_TYPE_CPU;
    case core::DeviceKind::Gpu:
        return GGML_BACKEND_DEVICE_TYPE_GPU;
    default:
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    }
}

static void ggml_arhat_query_devices(ggml_backend_reg *reg) { 
    int platform_count = core::Platform::Count();
    int num_devices = 0;
    for (int platform_id = 0; platform_id < platform_count; platform_id++) {
        core::Platform *platform = core::Platform::Get(platform_id);
        num_devices += platform->DeviceCount();
    }
    g_ggml_backend_arhat_devices.resize(num_devices);
    int index = 0;
    for (int platform_id = 0; platform_id < platform_count; platform_id++) {
        core::Platform *platform = core::Platform::Get(platform_id);
        int device_count = platform->DeviceCount();
        for (int device_id = 0; device_id < device_count; device_id++) {
            core::Device *device = platform->GetDevice(device_id);
            auto dev_ctx = new ggml_backend_arhat_device_context {
                /* .platform_id =        */ platform_id,
                /* .platform_name =      */ platform->Name(),
                /* .device_id   =        */ device_id,
                /* .device_type =        */ map_device_kind(device->Kind()),
                /* .device_name =        */ device->Name(),
                /* .device_description = */ device->Description(),
                /* .backend_ctx =        */ nullptr,
                /* .buffer_type =        */ {}
            };
            g_ggml_backend_arhat_devices[device_id] = {
                /* .iface   = */ ggml_backend_arhat_device_interface,
                /* .reg     = */ reg,
                /* .context = */ dev_ctx
            };
        }
    }
}

static const char *ggml_backend_arhat_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "Arhat";
} 

static size_t ggml_backend_arhat_reg_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return g_ggml_backend_arhat_devices.size();
} 

static ggml_backend_dev_t ggml_backend_arhat_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index < ggml_backend_arhat_reg_device_count(reg));
    return &g_ggml_backend_arhat_devices[index];
} 

static struct ggml_backend_reg_i ggml_backend_arhat_reg_interface = {
    /* .get_name         = */ ggml_backend_arhat_reg_get_name,
    /* .device_count     = */ ggml_backend_arhat_reg_device_count,
    /* .device_get       = */ ggml_backend_arhat_reg_device_get,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_arhat_reg(void) {
    static std::mutex mutex;
    static ggml_backend_reg reg;
    static bool initialized = false;
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return &reg;
    }
    initialized = true;
    arhat::onednn::base::Setup();
    arhat::onednn::gpu::Setup();
    ggml_arhat_init_op_map();
    ggml_arhat_query_devices(&reg); 
    reg = ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_arhat_reg_interface,
        /* .context     = */ NULL,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_arhat_reg)

