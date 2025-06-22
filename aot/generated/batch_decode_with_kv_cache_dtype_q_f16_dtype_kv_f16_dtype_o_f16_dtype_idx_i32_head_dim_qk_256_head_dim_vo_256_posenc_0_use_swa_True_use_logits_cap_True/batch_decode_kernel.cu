#include <flashinfer/attention/decode.cuh>
#include "batch_decode_config.inc"

using namespace flashinfer;

namespace flashinfer {

template cudaError_t
BatchDecodeWithPagedKVCacheDispatched<256, PosEncodingMode::kNone, DefaultAttention<false, true, true, false>, Params>(
    Params params, half* tmp_v,
    float* tmp_s, cudaStream_t stream);

};