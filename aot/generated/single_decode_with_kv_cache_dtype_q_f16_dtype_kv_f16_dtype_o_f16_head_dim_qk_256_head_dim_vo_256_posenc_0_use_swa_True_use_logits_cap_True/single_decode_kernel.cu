#include <flashinfer/attention/decode.cuh>
#include "single_decode_config.inc"

using namespace flashinfer;

namespace flashinfer {

template cudaError_t SingleDecodeWithKVCacheDispatched<
    256, PosEncodingMode::kNone, DefaultAttention<false, true, true, false>, Params>(
    Params params, half* tmp,
    cudaStream_t stream);

};