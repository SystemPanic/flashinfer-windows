#pragma once
#include <flashinfer/page.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/attention/variant_helper.cuh>
#include <flashinfer/attention/mla_params.cuh>

using namespace flashinfer;

#ifdef FLASHINFER_ENABLE_PROFILER
#define ADDITIONAL_FUNC_PARAMS , at::Tensor profiler_buffer
#define ADDITIONAL_PARAMS_SETTER \
  params.profiler_buffer = static_cast<uint64_t*>(profiler_buffer.data_ptr());
#else
#define ADDITIONAL_FUNC_PARAMS
#define ADDITIONAL_PARAMS_SETTER
#endif

using DTypeQ = half;
using DTypeKV = half;
using DTypeO = half;
using IdType = int32_t;
constexpr int HEAD_DIM_CKV = 512;
constexpr int HEAD_DIM_KPE = 64;

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE, Params, ...) \
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { \
    using Params = MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>; \
    __VA_ARGS__(); \
  })