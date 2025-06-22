#include <flashinfer/attention/prefill.cuh>
#include "batch_prefill_config.inc"

namespace flashinfer {

constexpr auto use_custom_mask = MaskMode::kCustom == MaskMode::kCustom;


template cudaError_t BatchPrefillWithRaggedKVCacheDispatched<
    /*CTA_TILE_Q=*/16, 256, 256, PosEncodingMode::kNone, false, MaskMode::kCustom,
    DefaultAttention<use_custom_mask, true, true, false>, RaggedParams>(RaggedParams params, half* tmp_v, float* tmp_s, cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched<
    /*CTA_TILE_Q=*/64, 256, 256, PosEncodingMode::kNone, false, MaskMode::kCustom,
    DefaultAttention<use_custom_mask, true, true, false>, RaggedParams>(RaggedParams params, half* tmp_v, float* tmp_s, cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched<
    /*CTA_TILE_Q=*/128, 256, 256, PosEncodingMode::kNone, false, MaskMode::kCustom,
    DefaultAttention<use_custom_mask, true, true, false>, RaggedParams>(RaggedParams params, half* tmp_v, float* tmp_s, cudaStream_t stream);


};  // namespace flashinfer