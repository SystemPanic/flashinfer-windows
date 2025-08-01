/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <cutlass/cutlass.h>

#include <cmath>
#include <cstdint>
#include <cute/tensor.hpp>

#include "../common.h"
#include "fmhaRunnerParams.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
using Dtype = Data_type;

struct KernelParams {
  // TMA descriptor for Q.
  CUtensorMap tmaQ_;
  // TMA descriptor for K.
  CUtensorMap tmaK_;
  // TMA descriptor for V.
  CUtensorMap tmaV_;
  // The descriptor for O.
  CUtensorMap tmaO_;

  // For FP4 KV cache, additional scaling factors are needed.
  // TMA descriptor for K scaling factor.
  CUtensorMap tmaKSf_;
  // TMA descriptor for V scaling factor.
  CUtensorMap tmaVSf_;

  // grid dimensions, these might differ from actual grid the kernel is launched with
  // for persistent kernels on Hopper GPUs.
  int32_t logicalGridDimX, logicalGridDimY, logicalGridDimZ;

  // The output pointer (used by STG for last tile).
  void* ptrO;
  // The output SF pointer (used for FP4 output).
  void* ptrSfO;

  // The cumulative sequence lengths for Q.
  int32_t const* ptrCumSeqLensQ;
  // The cumulative sequence lengths for K/V.
  int32_t const* ptrCumSeqLensKv;
  // The packed custom mask.
  uint32_t const* ptrCustomMask;
  // The packed custom mask's offsets of each sequence.
  int64_t const* ptrCustomMaskOffsets;
  // The debug output matrix O
  float* ptrDebugO;
  // The first sparseMask offsets in the Kv sequence dimension.
  int32_t const* ptrFirstSparseMaskOffsetsKv;
  // The counter for the multiCtasKv mode.
  int32_t* ptrMultiCtasKvCounter;
  // The device output scale for FP8 quantization. Only needed by trt-llm fp8 kernels as the sca-
  // les have to be on the device currently.
  float const* ptrOutputScale;
  // The page indexes of the paged-kv buffer with shape of [batchSize, 2, maxNumPagesPerSeq].
  int32_t const* ptrPageIdxKv;
  // The partial matrix O for each CtaKv when the multiCtasKv mode is enabled.
  void* ptrPartialO;
  // The partial softmax stats (max/sum)for each CtaKv when the multiCtasKv mode is enabled.
  float2* ptrPartialStats;
  // The scaling factors for K.
  float const* ptrSageAttnSfsK;
  // The scaling factors for P.
  float const* ptrSageAttnSfsP;
  // The scaling factors for Q.
  float const* ptrSageAttnSfsQ;
  // The scaling factors for V.
  float const* ptrSageAttnSfsV;
  // The device scaling factor for softmax (multiplied by log2 to use faster exp2). Only needed by
  // trt-llm fp8 kernels as the scales have to be on the device currently.
  float const* ptrScaleSoftmaxLog2;
  // The SF scale for Kv on device. Only needed by trt-llm kernels as the scales have to be on the
  // device currently.
  float const* ptrScaleSfKv;
  // The SF scale for O on device. Only needed by trt-llm kernels as the scales have to be on the
  // device currently.
  float const* ptrScaleSfO;
  // The sequence lengths for K/V. Required by pagedKv kernels to avoid unnecessary computation
  // based on (ptrCumSeqLensKv[batchIdx + 1] - ptrCumSeqLensKv[batchIdx]).
  int32_t const* ptrSeqLensKv;
  // The softmax stats buffer.
  float2* ptrSoftmaxStats;

  // The attention window size for sliding window attention.
  int32_t mAttentionWindowSize;
  // The batch size
  int32_t mBatchSize;
  // The chunked attention size in log2.
  int32_t mChunkedAttentionSizeLog2;
  // The log of the Sage Attention block size for K.
  int32_t mLogNumEltsPerSageAttnBlkK;
  // The log of the Sage Attention block size for P.
  int32_t mLogNumEltsPerSageAttnBlkP;
  // The log of the Sage Attention block size for Q.
  int32_t mLogNumEltsPerSageAttnBlkQ;
  // The log of the Sage Attention block size for V.
  int32_t mLogNumEltsPerSageAttnBlkV;
  // The sequence lengths for Q and K/V.
  int32_t mMaxSeqLenQ, mMaxSeqLenKv;
  // The maximum number of CTAs for Q.
  int32_t mMaxNumCtasQ;
  // The maximum number of CTAs for K/V.
  int32_t mMaxNumCtasKv;
  // The maximum number of pages per sequence for paged-kv buffer.
  int32_t mMaxNumPagesPerSeqKv;
  // The number of heads for K/V.
  int32_t mNumHeadsKv;
  // The number of heads for Q.
  int32_t mNumHeadsQ;
  // The number of Q heads per K/V head (i.e. mNumHeadsQ / mNumHeadsKv).
  int32_t mNumHeadsQPerKv;
  // The hidden size of O.
  int64_t mNumHiddenEltsO;
  // The number of MTP tokens per sequence. Assume that all requests have the same numMtpTokens
  // without paddings.
  int32_t mNumMtpTokens;
  // The total number of pages in the paged-kv memory pool.
  int32_t mNumPagesInMemPool;
  // The output scale for FP8 quantization.
  float mOutputScale;
  // The scaling factor for softmax (multiplied by log2 to use faster exp2).
  float mScaleSoftmaxLog2;
  // The SF scale for Kv.
  float mScaleSfKv;
  // The SF scale for O.
  float mScaleSfO;
  // The start token index in SF tensor. Used for FP4 SF offset calculation in generation phase
  // kernel when inflight batching is enabled in TRT-LLM.
  int32_t mStartTokenIdxSfO;
  // The sum of sequence lengths for Q and K/V.
  int32_t mSumOfSeqLensQ, mSumOfSeqLensKv;

  // Create the TMA shape/stride for Q.
  template <class FmhaOptions>
  static auto makeTmaShapeStrideQ(FmhaOptions const& options, bool groupsHeadsQ, int32_t tileSizeQ,
                                  int32_t numEltsInClampedHeadDimQ) {
    //
    // The Q has shape of [numTokens * numHeadsQPerKv, numHeadsKv * 1, headDim]
    // when grouping headsQ, otherwise it would be [numTokens, numHeadsQPerKv * numHeadsKv,
    // headDim].

    // The number of grouped heads for the A matrix of MMA.
    int32_t numGroupedHeads{1};
    if (groupsHeadsQ) {
      numGroupedHeads = std::min(tileSizeQ, options.mNumHeadsQPerKv);
    }

    // The number of heads.
    int32_t numHeads{options.mNumHeadsQ};
    if (groupsHeadsQ) {
      numHeads /= numGroupedHeads;
    }
    // Make sure the math works.
    TORCH_CHECK(numHeads * numGroupedHeads == options.mNumHeadsQ, "internal error");

    // The number of tokens.
    int32_t numTokens{options.mSumOfSeqLensQ};

    // This maps to flattened TMA shape for Q: (headDim, numTokens, numHeads).
    auto shape = std::vector<uint64_t>{
        static_cast<uint64_t>(options.mHeadDimQk), static_cast<uint64_t>(numGroupedHeads),
        static_cast<uint64_t>(numHeads), static_cast<uint64_t>(numTokens)};

    // The hidden dimension when the tensor contains only Q (i.e. not QKV packed).
    int32_t const hiddenDimQ{options.mNumHeadsQ * options.mHeadDimQk};

    // The hidden dimension when the Q, K and V tensors are packed.
    int32_t hiddenDimQkv{hiddenDimQ};
    if (isPackedQkv(options.mQkvLayout)) {
      TORCH_CHECK(!groupsHeadsQ, "internal error");
      hiddenDimQkv += options.mNumHeadsKv * (options.mHeadDimQk + options.mHeadDimV);
    }

    // The stride between tokens.
    int32_t strideTokens{hiddenDimQkv};

    // The stride between heads.
    int32_t strideHeads{groupsHeadsQ ? numGroupedHeads * options.mHeadDimQk : options.mHeadDimQk};

    // The stride between grouped heads.
    int32_t strideGroupedHeads{options.mHeadDimQk};

    // Assemble the stride (1, strideTokens, strideHeads).
    // Swap the first two dimension as mentioned before.
    auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(strideGroupedHeads),
                                        static_cast<uint64_t>(strideHeads),
                                        static_cast<uint64_t>(strideTokens)};

    // The tile shape for TMA.
    auto tileShapes = std::vector<uint32_t>{static_cast<uint32_t>(numEltsInClampedHeadDimQ), 1, 1,
                                            static_cast<uint32_t>(tileSizeQ)};
    if (groupsHeadsQ) {
      if (isSpecDecodingGenerationKernel(options.mKernelType)) {
        TORCH_CHECK((tileSizeQ % numGroupedHeads == 0), "internal error");
        tileShapes = std::vector<uint32_t>{static_cast<uint32_t>(numEltsInClampedHeadDimQ),
                                           static_cast<uint32_t>(numGroupedHeads), 1,
                                           static_cast<uint32_t>(tileSizeQ / numGroupedHeads)};
      } else {
        tileShapes = std::vector<uint32_t>{static_cast<uint32_t>(numEltsInClampedHeadDimQ),
                                           static_cast<uint32_t>(tileSizeQ), 1, 1};
      }
    }

    return std::make_tuple(shape, stride, tileShapes);
  }

  // Create the TMA shape/stride for O.
  template <class FmhaOptions>
  static auto makeTmaShapeStrideO(FmhaOptions const& options) {
    //
    // TODO: refactor this as makeTmaShapeStrideQ when removing cutlass tma copy.
    //

    // The number of tokens.
    int32_t numTokens{options.mSumOfSeqLensQ};

    // The number of heads per K/V head.
    int32_t numHeadsQPerKv{options.mNumHeadsQPerKv};

    // The batch dimension.
    int32_t batchSize{1};

    // The cute tensor shape for Q/O: (numTokens, headDim, ((numHeadsKv, numHeadsQPerKv),
    // batchSize)). This maps to flattened TMA shape for Q/O: (headDim, numTokens, numHeadsKv.
    // numHeadsQPerKv, batchSize). Note that TMA descriptor expects the first dimension's stride to
    // be 1, so swap the first two dimension so that the headDim dimension comes first.
    auto shape = std::vector<uint64_t>{
        static_cast<uint64_t>(options.mHeadDimV), static_cast<uint64_t>(numTokens),
        static_cast<uint64_t>(options.mNumHeadsKv), static_cast<uint64_t>(numHeadsQPerKv),
        static_cast<uint64_t>(batchSize)};

    // The hidden dimension.
    int32_t const hiddenDimO{options.mNumHeadsQ * options.mHeadDimV};

    // The stride between tokens.
    int32_t strideTokens{hiddenDimO};

    // The stride between Q heads.
    int32_t strideHeadsQ{options.mNumHeadsKv * options.mHeadDimV};

    // The stride between sequences.
    int32_t strideBatch{0};

    // The stride in between K/V heads.
    int32_t strideHeadsKv{options.mHeadDimV};
    // Assemble the stride (strideTokens, 1, ((strideHeadsKv, strideHeadsQ), strideBatch)).
    // Swap the first two dimension as mentioned before.
    auto stride = std::vector<uint64_t>{
        1, static_cast<uint64_t>(strideTokens), static_cast<uint64_t>(strideHeadsKv),
        static_cast<uint64_t>(strideHeadsQ), static_cast<uint64_t>(strideBatch)};

    return std::make_tuple(shape, stride);
  }

  // Create the shape for K and V.
  template <class FmhaOptions>
  static auto makeShapeKv(FmhaOptions const& options, KernelParams const& params) {
    // The number of keys/vals. WARNING: The if/else-if are sorted by priority.
    int32_t numKeysVals{options.mMaxSeqLenKv};
    if (isPagedKv(options.mQkvLayout)) {
      numKeysVals = options.mNumTokensPerPage;
    } else if (isContiguousKv(options.mQkvLayout)) {
      numKeysVals = options.mMaxSeqLenCacheKv;
    } else {
      numKeysVals = options.mSumOfSeqLensKv;
    }

    // The number of heads per K/V head (packed in the sequence length for mGroupsHeadsQ).
    int32_t numHeadsKv{options.mNumHeadsKv};

    // The batch dimension. WARNING: The if/else-if are sorted by priority.
    int32_t batchSize{options.mBatchSize};
    if (isPagedKv(options.mQkvLayout)) {
      batchSize = params.mNumPagesInMemPool;
    } else if (isContiguousKv(options.mQkvLayout)) {
      batchSize = options.mBatchSize;
    } else {
      batchSize = 1;
    }

    // Return the number of keys and batch.
    return std::make_tuple(numKeysVals, numHeadsKv, batchSize);
  }

  // Compute the strides for K and V.
  template <class FmhaOptions>
  static auto makeStrideKv(FmhaOptions const& options, bool isK) {
    // The maximum headDim of K and V.
    // Note that contiguousKv or pagedKv will pad K and V to maxHeadDimKv.
    int32_t const maxHeadDimKv{std::max(options.mHeadDimQk, options.mHeadDimV)};
    // The hidden dimension for the keys/vals.
    int32_t const hiddenDimK{options.mNumHeadsKv * options.mHeadDimQk};
    int32_t const hiddenDimV{options.mNumHeadsKv * options.mHeadDimV};
    int32_t const maxHiddenDimKv{std::max(hiddenDimK, hiddenDimV)};
    // The hidden dimension when Q, K and V are packed together.
    int32_t const hiddenDimQkv{options.mNumHeadsQ * options.mHeadDimQk +
                               options.mNumHeadsKv * (options.mHeadDimQk + options.mHeadDimV)};

    // The stride between the different keys/vals.
    int32_t strideKeysVals{isK ? hiddenDimK : hiddenDimV};
    if (isPagedKv(options.mQkvLayout)) {
      strideKeysVals = maxHeadDimKv;
      // printf("options.mNumTokensPerPage: %d, maxHeadDimKv: %d strideKeysVals: %d\n",
      //        options.mNumTokensPerPage, maxHeadDimKv, strideKeysVals);
    } else if (isPackedQkv(options.mQkvLayout)) {
      strideKeysVals = hiddenDimQkv;
    } else if (isContiguousKv(options.mQkvLayout)) {
      strideKeysVals = maxHeadDimKv;
    }

    // The stride between heads.
    int32_t strideHeads{isK ? options.mHeadDimQk : options.mHeadDimV};
    if (isPagedKv(options.mQkvLayout)) {
      strideHeads = options.mNumTokensPerPage * maxHeadDimKv;
      // printf("options.mNumTokensPerPage: %d, maxHeadDimKv: %d strideHeads: %d\n",
      //        options.mNumTokensPerPage, maxHeadDimKv, strideHeads);
    } else if (isContiguousKv(options.mQkvLayout)) {
      strideHeads = options.mMaxSeqLenCacheKv * maxHeadDimKv;
    }

    // The stride between batch items. WARNING: The order of if/else-if matters.
    int32_t strideBatch{options.mMaxSeqLenKv * hiddenDimK};
    if (isPagedKv(options.mQkvLayout)) {
      strideBatch = options.mNumTokensPerPage * maxHiddenDimKv;
      // printf("options.mNumTokensPerPage: %d, maxHiddenDimKv: %d strideBatch: %d\n",
      //        options.mNumTokensPerPage, maxHiddenDimKv, strideBatch);
    } else if (isContiguousKv(options.mQkvLayout)) {
      strideBatch = 2 * options.mNumHeadsKv * options.mMaxSeqLenCacheKv * maxHeadDimKv;
    } else {
      // Always variable seqlens.
      strideBatch = 0;
    }

    // The 3 strides (the other ones are 1 and 0).
    // todo(Yingyi): strideBatch is correct?
    return std::make_tuple(strideKeysVals, strideHeads, strideBatch);
  }

  // Create the TMA shape/stride for K.
  template <class FmhaOptions>
  static auto makeTmaShapeStrideKv(FmhaOptions const& options, KernelParams const& params,
                                   Data_type dtypeKv, bool isK) {
    // The shape elements.
    auto [numKeys, numHeadsQPerKv, batchSize] = makeShapeKv(options, params);
    // The stride elements.
    auto [strideKeys, strideHeads, strideBatch] = makeStrideKv(options, isK);

    // printf(
    //     "numKeys: %d, numHeadsQPerKv: %d, batchSize: %d, strideKeys: %d, strideHeads: %d, "
    //     "strideBatch: %d\n",
    //     numKeys, numHeadsQPerKv, batchSize, strideKeys, strideHeads, strideBatch);

    // The headDim.
    // Note that contiguousKv or pagedKv will pad K and V to maxHeadDimKv.
    int32_t headDim = isK ? options.mHeadDimQk : options.mHeadDimV;
    if (isPagedKv(options.mQkvLayout) || isContiguousKv(options.mQkvLayout)) {
      headDim = std::max(options.mHeadDimQk, options.mHeadDimV);
    }

    // For K, the cute layout: (numKeys, headDim, ((numHeadsQPerKv, numHeadsKv),
    // batchSize)):(strideKeys, _1, _0, strideHeads, strideBatch). Cute swaps the first two
    // dimension (to make sure stride of first dimension is 1) and ignores the numHeadsQPerKv
    // dimension (it's stride is always 0). For V, the headDim dimension is already the first
    // dimension so no swapping is needed.

    // Therefore, the resulting TMA layout is 4D: (headDim, numKeys, numHeadsKv, batchSize):(1,
    // strideKeys, strideHeads, strideBatch)

    // Note that for FP4 KV input, elements are stored as uint8_t, each packs 2 FP4 elements.
    // The column index and strides needs to divide by 2.
    auto const colIdxDivisor = dtypeKv == DATA_TYPE_E2M1 ? 2 : 1;
    auto shape = std::vector<uint64_t>{
        static_cast<uint64_t>(headDim / colIdxDivisor), static_cast<uint64_t>(numKeys),
        static_cast<uint64_t>(options.mNumHeadsKv), static_cast<uint64_t>(batchSize)};
    auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(strideKeys / colIdxDivisor),
                                        static_cast<uint64_t>(strideHeads / colIdxDivisor),
                                        static_cast<uint64_t>(strideBatch / colIdxDivisor)};

    return std::make_tuple(shape, stride);
  }

  // Create the TMA shape/stride for KV scaling factors.
  template <class FmhaOptions>
  static auto makeTmaShapeStrideKvSf(FmhaOptions const& options, KernelParams const& params,
                                     bool isK) {
    // The shape elements.
    auto [numKeys, numHeadsQPerKv, batchSize] = makeShapeKv(options, params);
    // The stride elements.
    auto [strideKeys, strideHeads, strideBatch] = makeStrideKv(options, isK);

    // The headDim.
    // Note that contiguousKv or pagedKv will pad K and V to maxHeadDimKv.
    int32_t headDim = isK ? options.mHeadDimQk : options.mHeadDimV;
    if (isPagedKv(options.mQkvLayout) || isContiguousKv(options.mQkvLayout)) {
      headDim = std::max(options.mHeadDimQk, options.mHeadDimV);
    }
    // The number of elements per SF.
    int32_t NumEltsPerSf = 16;

    // The KV shape is: (headDim, numKeys, numHeadsKv, batchSize)
    // Therefore, the KV SF shape should be (headDim / NumEltsPerSf, numKeys, numHeadsKv,
    // batchSize). Considering the TMA requires box width to be multiple of 16B, without changing
    // the underlying layout, we reshape into (16, numKeys * headDim / NumEltsPerSf / 16,
    // numHeadsKv, batchSize)

    // Note that it only works for pagedKv layout.
    TORCH_CHECK(isPagedKv(options.mQkvLayout), "The qkvLayout is not supported.");

    auto shape = std::vector<uint64_t>{
        16, static_cast<uint64_t>(numKeys * headDim / NumEltsPerSf / 16),
        static_cast<uint64_t>(options.mNumHeadsKv), static_cast<uint64_t>(batchSize)};
    auto stride = std::vector<uint64_t>{1, 16, static_cast<uint64_t>(strideHeads / NumEltsPerSf),
                                        static_cast<uint64_t>(strideBatch / NumEltsPerSf)};

    return std::make_tuple(shape, stride);
  }

  // Prepare pointers for TMA descriptors.
  static std::tuple<void const*, void const*, void const*> getDevicePtrs(
      TllmGenFmhaRunnerParams const& runnerParams, int32_t bytesPerElt) {
    // Declare the q, k, v ptrs.
    void const *qPtr{runnerParams.qPtr}, *kPtr{runnerParams.kPtr}, *vPtr{runnerParams.vPtr};

    // Set Q, K and V pointer from packed QKV tensor.
    if (isPackedQkv(runnerParams.mQkvLayout)) {
      qPtr = runnerParams.qkvPtr;
      kPtr = reinterpret_cast<void const*>(reinterpret_cast<char const*>(runnerParams.qkvPtr) +
                                           runnerParams.mNumHeadsQ * runnerParams.mHeadDimQk *
                                               bytesPerElt);
      vPtr = reinterpret_cast<void const*>(reinterpret_cast<char const*>(runnerParams.qkvPtr) +
                                           (runnerParams.mNumHeadsQ + runnerParams.mNumHeadsKv) *
                                               runnerParams.mHeadDimQk * bytesPerElt);
    }
    // Set K and V pointer from pagedKv tensor.
    else if (isPagedKv(runnerParams.mQkvLayout)) {
      // Note that the offsets will be fully handled by the pageIdx buffer.
      kPtr = runnerParams.kvPtr;
      vPtr = runnerParams.kvPtr;
    }
    // Set K and V pointer from contiguousQAnddKv tensor.
    else if (isContiguousKv(runnerParams.mQkvLayout)) {
      kPtr = runnerParams.kvPtr;
      // The maximum headDim of K and V.
      // Note that contiguousKv or pagedKv will pad K and V to maxHeadDimKv.
      int32_t const maxHeadDimKv{std::max(runnerParams.mHeadDimQk, runnerParams.mHeadDimV)};
      vPtr = reinterpret_cast<void const*>(
          reinterpret_cast<char const*>(runnerParams.kvPtr) +
          runnerParams.mNumHeadsKv * runnerParams.mMaxSeqLenCacheKv * maxHeadDimKv * bytesPerElt);
    }

    // Return the pointers.
    return std::make_tuple(qPtr, kPtr, vPtr);
  }

  // Build tma descriptors.
  template <class FmhaOptions>
  static CUtensorMap buildNdTmaDescriptor(FmhaOptions const& options, Data_type dtypeElt,
                                          std::vector<uint64_t> const& shapes,
                                          std::vector<uint64_t> const& strides,
                                          std::vector<uint32_t> const& tileShapes, void* gmemAddr,
                                          bool swizzled = true) {
    CUtensorMap desc{};
    // The data type.
    CUtensorMapDataType tmaDataFormat;
    if (dtypeElt == DATA_TYPE_E2M1 || dtypeElt == DATA_TYPE_E4M3) {
      tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    } else if (dtypeElt == DATA_TYPE_FP16) {
      tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if (dtypeElt == DATA_TYPE_BF16) {
      tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else {
      TORCH_CHECK(false, "Unexpected dtype %d", static_cast<int32_t>(dtypeElt));
    }

    // The swizzle type.
    CUtensorMapSwizzle swizzleType;
    int32_t numBytesInLeadingDim = tileShapes[0] * get_size_in_bits(dtypeElt) / 8 /*bits*/;
    if (!swizzled) {
      swizzleType = CU_TENSOR_MAP_SWIZZLE_NONE;
    } else if ((numBytesInLeadingDim % 128) == 0) {
      swizzleType = CU_TENSOR_MAP_SWIZZLE_128B;
    } else if ((numBytesInLeadingDim % 64) == 0) {
      swizzleType = CU_TENSOR_MAP_SWIZZLE_64B;
    } else if ((numBytesInLeadingDim % 32) == 0) {
      swizzleType = CU_TENSOR_MAP_SWIZZLE_32B;
    } else {
      TORCH_CHECK(false, "Unexpected numBytesInLeadingDim %d", numBytesInLeadingDim);
    }

    // Check gmem address must be 16B-aligned
    TORCH_CHECK((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0);

    // Check shape must be in range [1, 2^32]
    int32_t dim = shapes.size();
    // Max five dimension and min 3 dimension.
    TORCH_CHECK((dim <= 5) && (dim >= 3));
    // Check shape range.
    for (int32_t ii = 0; ii < dim; ++ii) {
      TORCH_CHECK(shapes[ii] >= (uint64_t(1)));        // Size must be min 1
      TORCH_CHECK(shapes[ii] <= (uint64_t(1) << 32));  // Size must be max 2^32
    }

    // TMA descriptor does not store the zeroth stride and assumes it is 1.
    TORCH_CHECK(static_cast<int32_t>(strides.size()) == dim);
    TORCH_CHECK(strides[0] == 1);

    // Build strides in bytes.
    // cuTensorMapEncodeTiled ignores the stride of the first dimension (implicitly 1).
    std::vector<uint64_t> stridesInBytes(dim - 1);
    for (int32_t ii = 0; ii < dim - 1; ++ii) {
      stridesInBytes[ii] = strides[ii + 1] *
                           std::max(get_size_in_bits(dtypeElt), static_cast<size_t>(8)) / 8 /*bit*/;
    }

    // Set tile strides to 0;
    std::vector<uint32_t> tileStrides(dim, 1);

    // Build the descriptor.
    CUresult result =
        cuTensorMapEncodeTiled(&desc, tmaDataFormat,
                               /*tensorRank=*/dim, gmemAddr, shapes.data(), stridesInBytes.data(),
                               tileShapes.data(), tileStrides.data(),
                               /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE, swizzleType,
                               /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS) {
      char const* err_str;
      cuGetErrorString(result, &err_str);
      std::cerr << "Error: Failed to initialize the TMA descriptor due to " << err_str << std::endl;
      std::cerr << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim
                << " gmem: " << gmemAddr << std::endl;
      std::cerr << "Shape: " << shapes[0] << " " << shapes[1] << " " << shapes[2] << " "
                << shapes[3] << " " << shapes[4] << std::endl;
      std::cerr << "Stride: " << stridesInBytes[0] << " " << stridesInBytes[1] << " "
                << stridesInBytes[2] << " " << stridesInBytes[3] << std::endl;
      std::cerr << "tileShapes: " << tileShapes[0] << " " << tileShapes[1] << " " << tileShapes[2]
                << " " << tileShapes[3] << " " << tileShapes[4] << std::endl;
      std::cerr << "tileStrides: " << tileStrides[0] << " " << tileStrides[1] << " "
                << tileStrides[2] << " " << tileStrides[3] << " " << tileStrides[4] << std::endl;
      std::cerr << "swizzleType: " << int(swizzleType) << std::endl;
      TORCH_CHECK(false);
    }

    return desc;
  }

  // Setup the kernel parameters.
  template <class FmhaOptions_, class KernelMeta>
  static KernelParams setKernelParams(FmhaOptions_ const& options, KernelMeta const& kernelMeta,
                                      int32_t maxNumCtasQ, int32_t maxNumCtasKv) {
    // Create the return struct.
    KernelParams params;

    // Get the device pointers for TMA descriptors.
    auto [qPtr, kPtr, vPtr] = getDevicePtrs(options, get_size_in_bytes(kernelMeta.mDataTypeKv));

    // The maximum headDim of K and V.
    // Note that contiguousKv or pagedKv will pad K and V to maxHeadDimKv.
    int32_t const maxHeadDimKv{std::max(options.mHeadDimQk, options.mHeadDimV)};

    // Set the number of pages in the memory pool for paged K/V cache.
    if (isPagedKv(options.mQkvLayout)) {
      params.mNumPagesInMemPool = options.mNumPagesInMemPool == 0
                                      ? options.mMaxNumPagesPerSeqKv * 2 * options.mBatchSize
                                      : options.mNumPagesInMemPool;
    }

    // The number of elements in 128B for Q.
    int32_t numEltsIn128BQ = (128 * 8) / get_size_in_bits(kernelMeta.mDataTypeQ);
    // The number of head elts (per token) in each block of shared memory.
    int32_t numEltsInClampedHeadDimQ = std::min(numEltsIn128BQ, options.mHeadDimQk);

    // Shape/stride for gmem tensor Q.
    auto [shapeQ, strideQ, tileShapeQ] = makeTmaShapeStrideQ(
        options, kernelMeta.mGroupsHeadsQ, kernelMeta.mTileSizeQ, numEltsInClampedHeadDimQ);
    // Build tma descriptor for Q.
    params.tmaQ_ = buildNdTmaDescriptor(options, kernelMeta.mDataTypeQ, shapeQ, strideQ, tileShapeQ,
                                        const_cast<void*>(qPtr));

    // The number of keys per tile.
    int32_t numKeysPerTile = isPagedKv(options.mQkvLayout)
                                 ? std::min(options.mNumTokensPerPage, kernelMeta.mTileSizeKv)
                                 : kernelMeta.mTileSizeKv;
    // The number of elements in 128B for Q.
    int32_t numEltsIn128BKv = (128 * 8) / get_size_in_bits(kernelMeta.mDataTypeKv);
    // The number of head elts (per token) in each block of shared memory (see above explanation).
    int32_t numEltsInClampedHeadDimKv = std::min(numEltsIn128BKv, maxHeadDimKv);

    // Shape/stride for gmem tensor Kv.
    auto [shapeK, strideK] =
        makeTmaShapeStrideKv(options, params, kernelMeta.mDataTypeKv, /*isK*/ true);
    auto [shapeV, strideV] =
        makeTmaShapeStrideKv(options, params, kernelMeta.mDataTypeKv, /*isK*/ false);
    // Build tma descriptor for K.
    // Do we have to transform K/V before MMA?
    bool const transformsKv{kernelMeta.mDataTypeKv != kernelMeta.mDataTypeQ};
    // Note that for FP4 KV input, elements are stored as uint8_t, each packs 2 FP4 elements.
    auto const numEltsDivisor = kernelMeta.mDataTypeKv == DATA_TYPE_E2M1 ? 2 : 1;
    // The tileShapes for K/V.
    std::vector<uint32_t> tileShapeKv(shapeK.size(), 1);
    tileShapeKv[0] = numEltsInClampedHeadDimKv / numEltsDivisor;
    tileShapeKv[1] = numKeysPerTile;
    // Build tma descriptor for K.
    params.tmaK_ = buildNdTmaDescriptor(options, kernelMeta.mDataTypeKv, shapeK, strideK,
                                        tileShapeKv, const_cast<void*>(kPtr),
                                        /*swizzled = */ !transformsKv);
    // Build tma descriptor for V.
    // todo(Yingyi): strideV is correct?
    params.tmaV_ = buildNdTmaDescriptor(options, kernelMeta.mDataTypeKv, shapeV, strideV,
                                        tileShapeKv, const_cast<void*>(vPtr),
                                        /*swizzled = */ !transformsKv);
    // params.tmaV_ = params.tmaK_;

    // If the KV dtype is E2m1, additional scaling factors are needed for dequant.
    if (kernelMeta.mDataTypeKv == DATA_TYPE_E2M1) {
      // The number of elements per SF.
      int32_t NumEltsPerSf = 16;
      // Compute the shape and stride for SF tensor.
      // FIXME: assume K and V uses the same shape.
      auto [shapeKvSf, strideKvSf] = makeTmaShapeStrideKvSf(options, params, /*isK*/ true);

      // The tileShapes for K/V.
      std::vector<uint32_t> tileShapeKvSf(shapeKvSf.size(), 1);
      tileShapeKvSf[0] = 16;
      tileShapeKvSf[1] = numKeysPerTile * maxHeadDimKv / NumEltsPerSf / 16;

      // The tile box is reshaped from (headDim / NumEltsPerSf, tileSizeKv) into (16, tileSizeKv *
      // headDim / NumEltsPerSf / 16). See makeTmaShapeStrideKvSf for details. Build tma descriptor
      // for K SF.
      params.tmaKSf_ = buildNdTmaDescriptor(options, DATA_TYPE_E4M3, shapeKvSf, strideKvSf,
                                            tileShapeKvSf, const_cast<void*>(options.kSfBasePtr),
                                            /*swizzled = */ false);

      // Build tma descriptor for V SF.
      params.tmaVSf_ = buildNdTmaDescriptor(options, DATA_TYPE_E4M3, shapeKvSf, strideKvSf,
                                            tileShapeKvSf, const_cast<void*>(options.vSfBasePtr),
                                            /*swizzled = */ false);
    }

    // Shape/stride for gmem tensor O.
    auto [shapeO, strideO] = makeTmaShapeStrideO(options);
    // The tileShapes for O.
    std::vector<uint32_t> tileShapeO(shapeO.size(), 1);
    tileShapeO[0] = numEltsInClampedHeadDimQ;
    tileShapeO[1] = kernelMeta.mTileSizeQ;
    // Build tma descriptor for O.
    params.tmaO_ = buildNdTmaDescriptor(options, kernelMeta.mDataTypeQ, shapeO, strideO, tileShapeO,
                                        const_cast<void*>(options.oPtr));

    // Set the other kernel parameters.
    params.ptrCumSeqLensQ = options.cumSeqLensQPtr;
    params.ptrCumSeqLensKv = options.cumSeqLensKvPtr;

    // The packed custom mask.
    params.ptrCustomMask = options.customMaskPtr;
    // The packed custom mask's offsets of each sequence.
    params.ptrCustomMaskOffsets = options.customMaskOffsetsPtr;
    // The first sparseMask offsets in the Kv sequence dimension.
    params.ptrFirstSparseMaskOffsetsKv = options.firstSparseMaskOffsetsKvPtr;

    // The output buffer.
    params.ptrO = options.oPtr;
    // The output scaling factor buffer.
    params.ptrSfO = options.oSfPtr;

    // TRT-LLM restrictions: the quantization scales must be on the device.
    params.ptrOutputScale = options.outputScalePtr;

    // The sequence lengths for Kv.
    params.ptrSeqLensKv = options.seqLensKvPtr;

    // The partial buffers' pointers when the multiCtasKv mode is enabled.
    int64_t partialStatsBufferSize = options.mMultiProcessorCount * kernelMeta.mStepQ;
    params.ptrMultiCtasKvCounter = options.multiCtasKvCounterPtr;
    params.ptrPartialStats = reinterpret_cast<float2*>(options.multiCtasKvScratchPtr);
    params.ptrPartialO = params.ptrPartialStats + partialStatsBufferSize;

    params.ptrPageIdxKv = options.kvPageIdxPtr;
    params.ptrScaleSoftmaxLog2 = options.scaleSoftmaxLog2Ptr;

    params.ptrScaleSfKv = options.kvSfScalePtr;
    params.ptrScaleSfO = options.oSfScalePtr;

    params.mAttentionWindowSize = options.mAttentionWindowSize;
    if (isSlidingOrChunkedCausalMask(
            static_cast<TrtllmGenAttentionMaskType>(kernelMeta.mMaskType)) &&
        options.mChunkedAttentionSize != INT_MAX) {
      TORCH_CHECK((options.mChunkedAttentionSize & (options.mChunkedAttentionSize - 1)) == 0,
                  "Chunked attention size must be a power of 2");
      params.mChunkedAttentionSizeLog2 = std::log2(options.mChunkedAttentionSize);
    } else {
      // Default 0 means that chunked attention is disabled.
      params.mChunkedAttentionSizeLog2 = 0;
    }
    params.mMaxSeqLenQ = options.mMaxSeqLenQ;
    params.mMaxSeqLenKv = options.mMaxSeqLenKv;
    params.mMaxNumCtasQ = maxNumCtasQ;
    params.mMaxNumCtasKv = maxNumCtasKv;
    params.mMaxNumPagesPerSeqKv = options.mMaxNumPagesPerSeqKv;
    // TODO: just use mMaxSeqLenQ for number of MTP tokens.
    params.mNumMtpTokens = options.mMaxSeqLenQ;
    params.mSumOfSeqLensQ = options.mSumOfSeqLensQ;
    params.mSumOfSeqLensKv = options.mSumOfSeqLensKv;
    params.mBatchSize = options.mBatchSize;
    params.mChunkedAttentionSizeLog2 = 0;
    params.mNumHeadsQ = options.mNumHeadsQ;
    params.mNumHeadsKv = options.mNumHeadsKv;
    params.mNumHeadsQPerKv = options.mNumHeadsQPerKv;
    params.mNumHiddenEltsO = options.mNumHeadsQ * options.mHeadDimQk;
    // todo(Yingyi): might take a scalar tensor later
    params.mOutputScale = options.outputScale;
    params.mScaleSoftmaxLog2 =
        (options.scaleSoftmaxLog2 / (std::sqrt((float)(options.mHeadDimQk)) * options.mScaleQ)) *
        M_LOG2E;
    params.mStartTokenIdxSfO = options.mSfStartTokenIdx;
    params.mScaleSfKv = options.mScaleSfKv;
    params.ptrSoftmaxStats = nullptr;

    return params;
  }
};
