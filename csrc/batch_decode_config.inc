/*
 * Copyright (c) 2025 by FlashInfer team.
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
// NOTE(Zihao): this is the include file for AOT mode
#pragma once
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

#include "aot_default_additional_params.h"
#include "aot_extension_utils.h"

using IdType = int32_t;

#define ADDITIONAL_FUNC_PARAMS BATCH_DECODE_ADDITIONAL_FUNC_PARAMS
#define ADDITIONAL_PARAMS_SETTER BATCH_DECODE_ADDITIONAL_PARAMS_SETTER

#define DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO,           \
                         POS_ENCODING_MODE, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP,          \
                         AttentionVariant, Params, ...)                                       \
  {                                                                                           \
    DISPATCH_PYTORCH_QKV_DTYPE_TO_CTYPE(q_scalar_type, kv_scalar_type, DTypeQ, DTypeKV, [&] { \
      using DTypeO = DTypeQ;                                                                  \
      using Params = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;                      \
      static constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;                       \
      return DISPATCH_head_dim(head_dim_qk, HEAD_DIM_QK, [&] {                                \
        [[maybe_unused]] static constexpr int HEAD_DIM_VO = HEAD_DIM_QK;                      \
        return DISPATCH_BOOL(window_left > -1, USE_SLIDING_WINDOW, [&] {                      \
          return DISPATCH_BOOL(logits_soft_cap > 0.f, USE_LOGITS_SOFT_CAP, [&] {              \
            static constexpr bool stat_use_slid_window = USE_SLIDING_WINDOW;                  \
            static constexpr bool stat_use_logits_soft_cap = USE_LOGITS_SOFT_CAP;             \
            using AttentionVariant =                                                          \
                DefaultAttention</*use_custom_mask=*/false, stat_use_slid_window,             \
                                 stat_use_logits_soft_cap, /*use_alibi_bias=*/false>;         \
            __VA_ARGS__();                                                                    \
            return true;                                                                      \
          });                                                                                 \
        });                                                                                   \
      });                                                                                     \
    });                                                                                       \
  }
