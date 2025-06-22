#include <flashinfer/attention/prefill.cuh>
#include "batch_prefill_config.inc"

namespace flashinfer {

constexpr auto use_custom_mask = MaskMode::kMultiItemScoring == MaskMode::kCustom;


template cudaError_t BatchPrefillWithRaggedKVCacheDispatched<
    /*CTA_TILE_Q=*/16, 128, 128, PosEncodingMode::kNone, false, MaskMode::kMultiItemScoring,
    DefaultAttention<use_custom_mask, false, false, false>, RaggedParams>(RaggedParams params, half* tmp_v, float* tmp_s, cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched<
    /*CTA_TILE_Q=*/64, 128, 128, PosEncodingMode::kNone, false, MaskMode::kMultiItemScoring,
    DefaultAttention<use_custom_mask, false, false, false>, RaggedParams>(RaggedParams params, half* tmp_v, float* tmp_s, cudaStream_t stream);

template cudaError_t BatchPrefillWithRaggedKVCacheDispatched<
    /*CTA_TILE_Q=*/128, 128, 128, PosEncodingMode::kNone, false, MaskMode::kMultiItemScoring,
    DefaultAttention<use_custom_mask, false, false, false>, RaggedParams>(RaggedParams params, half* tmp_v, float* tmp_s, cudaStream_t stream);


};  // namespace flashinfer