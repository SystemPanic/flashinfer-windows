#include <flashinfer/attention/prefill.cuh>
#include "single_prefill_config.inc"

using namespace flashinfer;

namespace flashinfer {

constexpr auto use_custom_mask = MaskMode::kMultiItemScoring == MaskMode::kCustom;

template cudaError_t SinglePrefillWithKVCacheDispatched<
    64, 64, PosEncodingMode::kNone, false, MaskMode::kMultiItemScoring, DefaultAttention<use_custom_mask, false, false, false>, Params>(
    Params params, half* tmp,
    cudaStream_t stream);

};