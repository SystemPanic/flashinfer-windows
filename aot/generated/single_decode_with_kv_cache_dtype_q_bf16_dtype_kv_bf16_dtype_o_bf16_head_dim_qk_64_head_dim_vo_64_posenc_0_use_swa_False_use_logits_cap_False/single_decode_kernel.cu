#include <flashinfer/attention/decode.cuh>
#include "single_decode_config.inc"

using namespace flashinfer;

namespace flashinfer {

template cudaError_t SingleDecodeWithKVCacheDispatched<
    64, PosEncodingMode::kNone, DefaultAttention<false, false, false, false>, Params>(
    Params params, nv_bfloat16* tmp,
    cudaStream_t stream);

};