#pragma once
#include <flashinfer/page.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/attention/variant_helper.cuh>

#define ADDITIONAL_FUNC_PARAMS , std::optional<at::Tensor> maybe_custom_mask, std::optional<at::Tensor> maybe_mask_indptr, std::optional<at::Tensor> maybe_alibi_slopes, std::optional<at::Tensor> maybe_prefix_len_ptr, std::optional<at::Tensor> maybe_token_pos_in_items_ptr, std::optional<at::Tensor> maybe_max_item_len_ptr, double logits_soft_cap, double sm_scale, double rope_rcp_scale, double rope_rcp_theta, int64_t token_pos_in_items_len
#define ADDITIONAL_PARAMS_SETTER params.maybe_custom_mask = maybe_custom_mask ? static_cast<uint8_t*>(maybe_custom_mask->data_ptr()): nullptr; \
params.maybe_mask_indptr = maybe_mask_indptr ? static_cast<int32_t*>(maybe_mask_indptr->data_ptr()): nullptr; \
params.maybe_alibi_slopes = maybe_alibi_slopes ? static_cast<float*>(maybe_alibi_slopes->data_ptr()): nullptr; \
params.maybe_prefix_len_ptr = maybe_prefix_len_ptr ? static_cast<uint32_t*>(maybe_prefix_len_ptr->data_ptr()): nullptr; \
params.maybe_token_pos_in_items_ptr = maybe_token_pos_in_items_ptr ? static_cast<uint16_t*>(maybe_token_pos_in_items_ptr->data_ptr()): nullptr; \
params.maybe_max_item_len_ptr = maybe_max_item_len_ptr ? static_cast<uint16_t*>(maybe_max_item_len_ptr->data_ptr()): nullptr; \
params.logits_soft_cap = logits_soft_cap; \
params.sm_scale = sm_scale; \
params.rope_rcp_scale = rope_rcp_scale; \
params.rope_rcp_theta = rope_rcp_theta; \
params.token_pos_in_items_len = token_pos_in_items_len;

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION, AttentionVariant, RaggedParams, PagedParams, ...) \
  DISPATCH_MASK_MODE(mask_mode, MASK_MODE, { \
    static constexpr auto use_custom_mask = MASK_MODE == MaskMode::kCustom; \
    using AttentionVariant = DefaultAttention<use_custom_mask, true, true, false>; \
    __VA_ARGS__(); \
  })

using namespace flashinfer;

using DTypeQ = half;
using DTypeKV = __nv_fp8_e4m3;
using DTypeO = half;
using IdType = int32_t;
static constexpr int HEAD_DIM_QK = 256;
static constexpr int HEAD_DIM_VO = 256;
static constexpr bool USE_FP16_QK_REDUCTION = false;
static constexpr auto USE_LOGITS_SOFT_CAP = true;
static constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;
static constexpr auto USE_SLIDING_WINDOW = true;


struct RaggedParams {
  using DTypeQ = DTypeQ;
  using DTypeKV = DTypeKV;
  using DTypeO = DTypeO;
  using IdType = IdType;

  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  IdType* q_indptr;
  IdType* kv_indptr;
  DTypeO* o;
  float* lse;
  uint_fastdiv group_size;

  uint8_t* maybe_custom_mask;
int32_t* maybe_mask_indptr;
float* maybe_alibi_slopes;
uint32_t* maybe_prefix_len_ptr;
uint16_t* maybe_token_pos_in_items_ptr;
uint16_t* maybe_max_item_len_ptr;
double logits_soft_cap;
double sm_scale;
double rope_rcp_scale;
double rope_rcp_theta;
int64_t token_pos_in_items_len;

  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t k_stride_n;
  uint32_t k_stride_h;
  uint32_t v_stride_n;
  uint32_t v_stride_h;
  int32_t window_left;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
  }
};

struct PagedParams {
  using DTypeQ = DTypeQ;
  using DTypeKV = DTypeKV;
  using DTypeO = DTypeO;
  using IdType = IdType;

  DTypeQ* q;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  IdType* q_indptr;
  DTypeO* o;
  float* lse;
  uint_fastdiv group_size;

  uint8_t* maybe_custom_mask;
int32_t* maybe_mask_indptr;
float* maybe_alibi_slopes;
uint32_t* maybe_prefix_len_ptr;
uint16_t* maybe_token_pos_in_items_ptr;
uint16_t* maybe_max_item_len_ptr;
double logits_soft_cap;
double sm_scale;
double rope_rcp_scale;
double rope_rcp_theta;
int64_t token_pos_in_items_len;

  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  int32_t window_left;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  bool* block_valid_mask;
  IdType* kv_chunk_size_ptr;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

#include<flashinfer/attention/variants.cuh>