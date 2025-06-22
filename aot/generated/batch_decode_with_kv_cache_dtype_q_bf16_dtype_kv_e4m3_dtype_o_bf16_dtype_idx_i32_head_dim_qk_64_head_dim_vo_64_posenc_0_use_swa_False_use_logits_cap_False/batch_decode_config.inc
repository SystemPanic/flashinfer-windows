#pragma once
#include <flashinfer/page.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/attention/variant_helper.cuh>

#define ADDITIONAL_FUNC_PARAMS , std::optional<at::Tensor> maybe_alibi_slopes, double logits_soft_cap, double sm_scale, double rope_rcp_scale, double rope_rcp_theta
#define ADDITIONAL_PARAMS_SETTER params.maybe_alibi_slopes = maybe_alibi_slopes ? static_cast<float*>(maybe_alibi_slopes->data_ptr()): nullptr; \
params.logits_soft_cap = logits_soft_cap; \
params.sm_scale = sm_scale; \
params.rope_rcp_scale = rope_rcp_scale; \
params.rope_rcp_theta = rope_rcp_theta;

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, ...) { \
  using AttentionVariant = DefaultAttention<false, false, false, false>; \
  __VA_ARGS__(); \
}

using namespace flashinfer;

using DTypeQ = nv_bfloat16;
using DTypeKV = __nv_fp8_e4m3;
using DTypeO = nv_bfloat16;
using IdType = int32_t;
static constexpr int HEAD_DIM_QK = 64;
static constexpr int HEAD_DIM_VO = 64;
static constexpr auto USE_LOGITS_SOFT_CAP = false;
static constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;
static constexpr auto USE_SLIDING_WINDOW = false;

struct Params {
  using DTypeQ = DTypeQ;
  using DTypeKV = DTypeKV;
  using DTypeO = DTypeO;
  using IdType = IdType;

  DTypeQ* q;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  DTypeO* o;
  float* lse;

  float* maybe_alibi_slopes;
double logits_soft_cap;
double sm_scale;
double rope_rcp_scale;
double rope_rcp_theta;


  uint32_t padded_batch_size;
  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  int32_t window_left;

  IdType* request_indices;
  IdType* kv_tile_indices;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  bool partition_kv;

  __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t batch_idx) const { return 1; }

  __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

#include<flashinfer/attention/variants.cuh>