"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ctypes
import functools
import os

# Re-export
from . import cubin_loader
from . import env as env
from .activation import gen_act_and_mul_module as gen_act_and_mul_module
from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
from .attention import gen_cudnn_fmha_module as gen_cudnn_fmha_module
from .attention import gen_batch_attention_module as gen_batch_attention_module
from .attention import gen_batch_decode_module as gen_batch_decode_module
from .attention import gen_batch_mla_module as gen_batch_mla_module
from .attention import gen_batch_prefill_module as gen_batch_prefill_module
from .attention import (
    gen_customize_batch_decode_module as gen_customize_batch_decode_module,
)
from .attention import (
    gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
)
from .attention import (
    gen_customize_single_decode_module as gen_customize_single_decode_module,
)
from .attention import (
    gen_customize_single_prefill_module as gen_customize_single_prefill_module,
)
from .attention import gen_fmha_cutlass_sm100a_module as gen_fmha_cutlass_sm100a_module
from .attention import gen_batch_pod_module as gen_batch_pod_module
from .attention import gen_pod_module as gen_pod_module
from .attention import gen_single_decode_module as gen_single_decode_module
from .attention import gen_single_prefill_module as gen_single_prefill_module
from .attention import get_batch_attention_uri as get_batch_attention_uri
from .attention import get_batch_decode_uri as get_batch_decode_uri
from .attention import get_batch_mla_uri as get_batch_mla_uri
from .attention import get_batch_prefill_uri as get_batch_prefill_uri
from .attention import get_pod_uri as get_pod_uri
from .attention import get_single_decode_uri as get_single_decode_uri
from .attention import get_single_prefill_uri as get_single_prefill_uri
from .attention import gen_trtllm_gen_fmha_module as gen_trtllm_gen_fmha_module
from .attention import gen_fmha_v2_module as gen_fmha_v2_module
from .attention import (
    gen_trtllm_fmha_v2_sm120_module as gen_trtllm_fmha_v2_sm120_module,
)
from .core import JitSpec as JitSpec
from .core import JitSpecNvcc as JitSpecNvcc
from .cute_dsl_core import JitSpecCuteDsl as JitSpecCuteDsl
from .core import JitSpecStatus as JitSpecStatus
from .core import JitSpecRegistry as JitSpecRegistry
from .core import jit_spec_registry as jit_spec_registry
from .core import build_jit_specs as build_jit_specs
from .core import clear_cache_dir as clear_cache_dir
from .core import gen_jit_spec as gen_jit_spec
from .core import MissingJITCacheError as MissingJITCacheError
from .core import sm90a_nvcc_flags as sm90a_nvcc_flags
from .core import sm100a_nvcc_flags as sm100a_nvcc_flags
from .core import sm100f_nvcc_flags as sm100f_nvcc_flags
from .core import sm103a_nvcc_flags as sm103a_nvcc_flags
from .core import sm107a_nvcc_flags as sm107a_nvcc_flags
from .core import sm110a_nvcc_flags as sm110a_nvcc_flags
from .core import sm120a_nvcc_flags as sm120a_nvcc_flags
from .core import sm120f_nvcc_flags as sm120f_nvcc_flags
from .core import sm121a_nvcc_flags as sm121a_nvcc_flags
from .core import current_compilation_context as current_compilation_context
from .cubin_loader import setup_cubin_loader
from .comm import gen_trtllm_mnnvl_comm_module as gen_trtllm_mnnvl_comm_module
from .comm import gen_trtllm_comm_module as gen_trtllm_comm_module
from .comm import gen_vllm_comm_module as gen_vllm_comm_module
from .comm import gen_pcie_ipc_comm_module as gen_pcie_ipc_comm_module
from .comm import gen_cake_alltoall_module as gen_cake_alltoall_module
from .comm import (
    gen_nvlink_one_sided_alltoall_module as gen_nvlink_one_sided_alltoall_module,
)
from .comm import (
    gen_nvlink_two_sided_alltoall_module as gen_nvlink_two_sided_alltoall_module,
)
from .comm import gen_dcp_alltoall_module as gen_dcp_alltoall_module
from .dsv3_optimizations import (
    gen_dsv3_router_gemm_module as gen_dsv3_router_gemm_module,
)
from .dsv3_optimizations import (
    gen_dsv3_fused_routing_module as gen_dsv3_fused_routing_module,
)
from .tinygemm2 import gen_tinygemm2_module as gen_tinygemm2_module
from .tinygemm2 import gen_tinygemm2_sm100_module as gen_tinygemm2_sm100_module
from .cake_megamoe_topk_reduce import (
    gen_cake_megamoe_topk_reduce_module as gen_cake_megamoe_topk_reduce_module,
)
from .cake_sampling import gen_cake_sampling_module as gen_cake_sampling_module
from .cake_sampling import get_cake_sampling_uri as get_cake_sampling_uri
from .cake_sampling import load_cake_sampling_module as load_cake_sampling_module
from .cake_megamoe_topk_reduce import (
    get_cake_megamoe_topk_reduce_module as get_cake_megamoe_topk_reduce_module,
)
from .cake_megamoe_topk_reduce import (
    get_cake_megamoe_topk_reduce_uri as get_cake_megamoe_topk_reduce_uri,
)
from .cake_megamoe_topk_reduce import (
    load_cake_megamoe_topk_reduce_module as load_cake_megamoe_topk_reduce_module,
)
from .cake_megamoe_topk_reduce import (
    is_cake_megamoe_topk_reduce_module_loaded as is_cake_megamoe_topk_reduce_module_loaded,
)
from .cake_megamoe_topk_reduce import (
    run_cake_megamoe_topk_reduce as run_cake_megamoe_topk_reduce,
)
from .fused_moe import gen_alphamoe_sm100_module as gen_alphamoe_sm100_module
from .moe_utils import gen_moe_utils_module as gen_moe_utils_module
from .hash_topk import gen_hash_topk_module as gen_hash_topk_module
from .q_token_kv_block_sparse_metadata import (
    gen_prims_ts_q_token_kv_block_sparse_metadata_module as gen_prims_ts_q_token_kv_block_sparse_metadata_module,
)
from .fp4_kv_dequantization import (
    gen_fp4_kv_dequantization_module as gen_fp4_kv_dequantization_module,
)
from .fp4_kv_quantization import (
    gen_fp4_kv_quantization_module as gen_fp4_kv_quantization_module,
)
from .cake_kda import (
    cake_kda_affine_is_available as cake_kda_affine_is_available,
)
from .cake_kda import (
    gen_cake_kda_affine_module as gen_cake_kda_affine_module,
)
from .cake_kda import (
    gen_cake_kda_m128_unbounded_softplus_module as gen_cake_kda_m128_unbounded_softplus_module,
)
from .cake_kda import (
    get_cake_kda_affine_module as get_cake_kda_affine_module,
)
from .cake_kda import (
    get_cake_kda_prefill_module as get_cake_kda_prefill_module,
)
from .cake_kda import (
    load_cake_kda_m128_unbounded_softplus_module as load_cake_kda_m128_unbounded_softplus_module,
)
from .cake_fused_moe_warp_decode import (
    gen_cake_fused_moe_warp_decode_module as gen_cake_fused_moe_warp_decode_module,
)
from .cake_fused_moe_warp_decode import (
    get_cake_fused_moe_warp_decode_module as get_cake_fused_moe_warp_decode_module,
)
from .flash_kda import (
    gen_flash_kda_m64_module as gen_flash_kda_m64_module,
)
from .flash_kda import (
    gen_flash_kda_m128_module as gen_flash_kda_m128_module,
)
from .flash_kda import (
    gen_flash_kda_m128_n16_module as gen_flash_kda_m128_n16_module,
)
from .flash_kda import (
    get_flash_kda_prefill_module as get_flash_kda_prefill_module,
)
from .flash_kda import (
    load_flash_kda_m64_module as load_flash_kda_m64_module,
)
from .flash_kda import (
    load_flash_kda_m128_module as load_flash_kda_m128_module,
)
from .flash_kda import (
    load_flash_kda_m128_n16_module as load_flash_kda_m128_n16_module,
)
from .blackwell_msa import BLACKWELL_MSA_VARIANTS as BLACKWELL_MSA_VARIANTS
from .blackwell_msa import (
    BLACKWELL_MSA_VARIANTS_BY_TARGET as BLACKWELL_MSA_VARIANTS_BY_TARGET,
)
from .blackwell_msa import gen_blackwell_msa_module as gen_blackwell_msa_module
from .blackwell_msa import get_blackwell_msa_module as get_blackwell_msa_module
from .blackwell_msa import get_blackwell_msa_uri as get_blackwell_msa_uri
from .blackwell_msa import load_blackwell_msa_module as load_blackwell_msa_module
from .nvfp4_attention_sm120 import (
    gen_nvfp4_attention_sm120_module as gen_nvfp4_attention_sm120_module,
)
from .bgmv_moe import gen_bgmv_moe_module as gen_bgmv_moe_module
from .bgmv_moe import load_bgmv_moe_module as load_bgmv_moe_module
from .cake_bgmv_moe import (
    gen_cake_bgmv_moe_module as gen_cake_bgmv_moe_module,
)
from .cake_bgmv_moe import (
    load_cake_bgmv_moe_module as load_cake_bgmv_moe_module,
)
from .monomoe import gen_monomoe_module as gen_monomoe_module
from .monomoe import load_monomoe_module as load_monomoe_module
from .fused_moe import (
    gen_alphamoe_fused_router_module as gen_alphamoe_fused_router_module,
)
from .fused_moe import (
    gen_alphamoe_nvfp4_sm100_module as gen_alphamoe_nvfp4_sm100_module,
)

def _load_windows_cudart() -> None:
    import glob
    import sysconfig

    import torch

    # Use the *Python process* architecture, not the host CPU: an x64 Python
    # running under emulation on an ARM64 machine still needs x64 DLLs.
    plat = sysconfig.get_platform().lower()  # e.g. "win-arm64", "win-amd64"
    is_arm64 = plat.endswith("arm64") or plat.endswith("aarch64")
    arch = "arm64" if is_arm64 else "x64"

    cuda_path = None
    for var in ("CUDA_HOME", "CUDA_ROOT", "CUDA_PATH"):
        if os.environ.get(var):
            cuda_path = os.environ.get(var)
            break
    else:
        if os.environ.get("CUDA_LIB_PATH"):
            # CUDA_LIB_PATH usually points at <root>\lib\x64 or <root>\lib\arm64
            cuda_path = os.path.abspath(
                os.path.join(os.environ.get("CUDA_LIB_PATH"), "..", "..")
            )
        else:
            cuda_path = (
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\"
                f"v{torch.version.cuda}"
            )

    if not cuda_path or not os.path.exists(cuda_path):
        raise ValueError(
            "CUDA toolkit not found. Set CUDA_PATH (or CUDA_HOME / CUDA_ROOT) "
            "to the CUDA root folder, or CUDA_LIB_PATH to its lib\\<arch> "
            "folder (for example, set CUDA_PATH=C:\\CUDA\\v12.4)"
        )

    subdirs = ("bin\\arm64", "bin\\aarch64", "bin") if is_arm64 else ("bin\\x64", "bin")
    bin_dirs = [
        os.path.join(cuda_path, d)
        for d in subdirs
        if os.path.isdir(os.path.join(cuda_path, d))
    ]

    major = int(torch.version.cuda.split(".")[0])
    # CUDA >= 12: cudart64_12.dll; CUDA 11: cudart64_110.dll
    suffix = str(major) if major >= 12 else f"{major}0"
    names = (f"cudart64_{suffix}.dll", f"cudart_{suffix}.dll")

    for bin_dir in bin_dirs:
        candidates = [os.path.join(bin_dir, n) for n in names]
        # Fallback: whatever cudart DLL the toolkit ships for this arch
        candidates += sorted(glob.glob(os.path.join(bin_dir, "cudart*.dll")))
        for dll in candidates:
            if os.path.isfile(dll):
                # Let dependent CUDA DLLs resolve from the same directory
                os.add_dll_directory(bin_dir)
                ctypes.CDLL(dll, mode=ctypes.RTLD_GLOBAL)
                return

    raise FileNotFoundError(
        f"Could not find a {arch} cudart DLL for CUDA {torch.version.cuda} "
        f"under {cuda_path} (searched: {', '.join(bin_dirs) or 'no bin dirs'})"
    )


def _default_linux_cuda_lib_path() -> str:
    machine = os.uname().machine.lower()
    if machine in ("aarch64", "arm64"):
        for target in ("sbsa-linux", "aarch64-linux"):
            path = f"/usr/local/cuda/targets/{target}/lib/"
            if os.path.isdir(path):
                return path
        return "/usr/local/cuda/targets/sbsa-linux/lib/"
    return "/usr/local/cuda/targets/x86_64-linux/lib/"

if os.name == "nt":
    _load_windows_cudart()
else:
    cuda_lib_path = os.environ.get("CUDA_LIB_PATH", _default_linux_cuda_lib_path())
    if os.path.exists(f"{cuda_lib_path}/libcudart.so.12"):
        ctypes.CDLL(f"{cuda_lib_path}/libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
