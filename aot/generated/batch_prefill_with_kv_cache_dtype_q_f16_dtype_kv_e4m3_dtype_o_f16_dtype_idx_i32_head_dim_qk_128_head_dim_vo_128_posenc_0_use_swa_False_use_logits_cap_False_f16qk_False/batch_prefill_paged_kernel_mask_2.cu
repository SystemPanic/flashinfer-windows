#include <flashinfer/attention/prefill.cuh>
#include "batch_prefill_config.inc"

namespace flashinfer {

constexpr auto use_custom_mask = MaskMode::kCustom == MaskMode::kCustom;


template cudaError_t BatchPrefillWithPagedKVCacheDispatched<
    /*CTA_TILE_Q=*/16, 128, 128, PosEncodingMode::kNone, false, MaskMode::kCustom,
    DefaultAttention<use_custom_mask, false, false, false>, PagedParams>(PagedParams params, half* tmp_v, float* tmp_s, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<
    /*CTA_TILE_Q=*/64, 128, 128, PosEncodingMode::kNone, false, MaskMode::kCustom,
    DefaultAttention<use_custom_mask, false, false, false>, PagedParams>(PagedParams params, half* tmp_v, float* tmp_s, cudaStream_t stream);

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<
    /*CTA_TILE_Q=*/128, 128, 128, PosEncodingMode::kNone, false, MaskMode::kCustom,
    DefaultAttention<use_custom_mask, false, false, false>, PagedParams>(PagedParams params, half* tmp_v, float* tmp_s, cudaStream_t stream);


};  // namespace flashinfer