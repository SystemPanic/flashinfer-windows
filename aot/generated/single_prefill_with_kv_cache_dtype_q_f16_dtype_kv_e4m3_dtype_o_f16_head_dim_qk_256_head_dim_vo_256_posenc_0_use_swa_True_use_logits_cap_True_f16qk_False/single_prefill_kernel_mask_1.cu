#include <flashinfer/attention/prefill.cuh>
#include "single_prefill_config.inc"

using namespace flashinfer;

namespace flashinfer {

constexpr auto use_custom_mask = MaskMode::kCausal == MaskMode::kCustom;

template cudaError_t SinglePrefillWithKVCacheDispatched<
    256, 256, PosEncodingMode::kNone, false, MaskMode::kCausal, DefaultAttention<use_custom_mask, true, true, false>, Params>(
    Params params, half* tmp,
    cudaStream_t stream);

};