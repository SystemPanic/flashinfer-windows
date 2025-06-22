
#include <flashinfer/activation.cuh>
#include "pytorch_extension_utils.h"
#include <cuda_runtime.h>



using namespace flashinfer;


__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}


void silu_and_mul(at::Tensor& out, at::Tensor& input, bool enable_pdl) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const c10::cuda::OptionalCUDAGuard device_guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = flashinfer::activation::act_and_mul_kernel<c_type, silu>;

    cudaLaunchKernelEx(&config, kernel, static_cast<c_type*>(out.data_ptr()),
                       static_cast<c_type*>(input.data_ptr()), d);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Failed to launch kernel: ", cudaGetErrorString(err));

    return true;
  });
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul", silu_and_mul);
}