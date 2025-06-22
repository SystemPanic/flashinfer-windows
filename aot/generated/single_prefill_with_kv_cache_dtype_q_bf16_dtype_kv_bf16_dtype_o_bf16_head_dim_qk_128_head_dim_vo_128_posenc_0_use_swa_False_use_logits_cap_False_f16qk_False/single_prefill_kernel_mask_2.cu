#include <flashinfer/attention/prefill.cuh>
#include "single_prefill_config.inc"

using namespace flashinfer;

namespace flashinfer {

constexpr auto use_custom_mask = MaskMode::kCustom == MaskMode::kCustom;

template cudaError_t SinglePrefillWithKVCacheDispatched<
    128, 128, PosEncodingMode::kNone, false, MaskMode::kCustom, DefaultAttention<use_custom_mask, false, false, false>, Params>(
    Params params, nv_bfloat16* tmp,
    cudaStream_t stream);

};