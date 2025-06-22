#include <flashinfer/attention/prefill.cuh>
#include "batch_prefill_config.inc"

namespace flashinfer {

constexpr auto use_custom_mask = MaskMode::kCustom == MaskMode::kCustom;


template cudaError_t BatchPrefillWithPagedKVCacheDispatched<
    /*CTA_TILE_Q=*/16, 64, 64, PosEncodingMode::kNone, false, MaskMode::kCustom,
    DefaultAttention<use_custom_mask, false, false, false>, PagedParams>(PagedParams params, nv_bfloat16* tmp_v, float* tmp_s, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<
    /*CTA_TILE_Q=*/64, 64, 64, PosEncodingMode::kNone, false, MaskMode::kCustom,
    DefaultAttention<use_custom_mask, false, false, false>, PagedParams>(PagedParams params, nv_bfloat16* tmp_v, float* tmp_s, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<
    /*CTA_TILE_Q=*/128, 64, 64, PosEncodingMode::kNone, false, MaskMode::kCustom,
    DefaultAttention<use_custom_mask, false, false, false>, PagedParams>(PagedParams params, nv_bfloat16* tmp_v, float* tmp_s, cudaStream_t stream);


};  // namespace flashinfer