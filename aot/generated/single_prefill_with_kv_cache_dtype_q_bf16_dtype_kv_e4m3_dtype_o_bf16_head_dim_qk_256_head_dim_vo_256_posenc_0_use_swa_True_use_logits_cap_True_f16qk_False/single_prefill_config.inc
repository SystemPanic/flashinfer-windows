#pragma once
#include <flashinfer/math.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/attention/variant_helper.cuh>

#define ADDITIONAL_FUNC_PARAMS , std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_alibi_slopes, double logits_soft_cap, double sm_scale, double rope_rcp_scale, double rope_rcp_theta
#define ADDITIONAL_PARAMS_SETTER params.maybe_custom_mask = maybe_custom_mask ? static_cast<uint8_t*>(maybe_custom_mask->data_ptr()): nullptr; \
params.maybe_alibi_slopes = maybe_alibi_slopes ? static_cast<float*>(maybe_alibi_slopes->data_ptr()): nullptr; \
params.logits_soft_cap = logits_soft_cap; \
params.sm_scale = sm_scale; \
params.rope_rcp_scale = rope_rcp_scale; \
params.rope_rcp_theta = rope_rcp_theta;


#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant, Params, ...) \
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { \
    static constexpr bool use_custom_mask = MASK_MODE == MaskMode::kCustom; \
    using AttentionVariant = DefaultAttention<use_custom_mask, true, true, false>; \
    __VA_ARGS__(); \
  })


using namespace flashinfer;

using DTypeQ = nv_bfloat16;
using DTypeKV = __nv_fp8_e4m3;
using DTypeO = nv_bfloat16;
using IdType = int32_t;
static constexpr int HEAD_DIM_QK = 256;
static constexpr int HEAD_DIM_VO = 256;
static constexpr bool USE_FP16_QK_REDUCTION = false;
static constexpr auto USE_LOGITS_SOFT_CAP = true;
static constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;
static constexpr auto USE_SLIDING_WINDOW = true;

struct Params {
  using DTypeQ = DTypeQ;
  using DTypeKV = DTypeKV;
  using DTypeO = DTypeO;
  using IdType = int32_t;
  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  DTypeO* o;
  float* lse;
  uint_fastdiv group_size;

  uint8_t* maybe_custom_mask;
float* maybe_alibi_slopes;
double logits_soft_cap;
double sm_scale;
double rope_rcp_scale;
double rope_rcp_theta;


  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t k_stride_n;
  uint32_t k_stride_h;
  uint32_t v_stride_n;
  uint32_t v_stride_h;
  uint32_t head_dim;
  int32_t window_left;

  bool partition_kv;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return qo_len;
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_len;
  }
};

#include<flashinfer/attention/variants.cuh>