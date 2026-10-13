/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>

#include "batch_prefill_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {

template <bool SAME_KV_STRIDES, uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, bool enable_pdl,
                                                   cudaStream_t stream);

}  // namespace flashinfer

// Windows Fix: Must mirror the CTA_TILE_Q list in batch_prefill_{paged,ragged}_kernel_inst.jinja.
// DISPATCH_CTA_TILE_Q expands every case (incl. 32), so without this guard the
// dispatcher references template instances that are never emitted. GCC/ld tolerate
// unresolved symbols in a .so, but MSVC link.exe fails with LNK2019/LNK1120.
template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO>
inline constexpr bool kCtaTileQInstantiated =
    (HEAD_DIM_VO >= 512)   ? (CTA_TILE_Q == 16 || CTA_TILE_Q == 32)
    : (HEAD_DIM_QK >= 512) ? (CTA_TILE_Q == 16)
                           : (CTA_TILE_Q == 16 || CTA_TILE_Q == 64 || CTA_TILE_Q == 128);

using namespace flashinfer;

using tvm::ffi::Array;
using tvm::ffi::Optional;

#include "batch_prefill_paged.cuh"
