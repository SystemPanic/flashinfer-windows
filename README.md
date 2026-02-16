<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-black-background.png?raw=true">
    <img alt="FlashInfer" src="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-white-background.png?raw=true" width=55%>
  </picture>
</p>
<h1 align="center">
High-Performance GPU Kernels for Inference
</h1>

<p align="center">
| <a href="https://docs.flashinfer.ai"><b>Documentation</b></a> | <a href="https://github.com/flashinfer-ai/flashinfer/releases/latest"><b>Latest Release</b></a> | <a href="https://flashinfer.ai"><b>Blog</b></a> | <a href="https://join.slack.com/t/flashinfer/shared_invite/zt-379wct3hc-D5jR~1ZKQcU00WHsXhgvtA"><b>Slack</b></a> |  <a href="https://github.com/orgs/flashinfer-ai/discussions"><b>Discussion Forum</b></a> |
</p>

[![Build Status](https://ci.tlcpack.ai/job/flashinfer-ci/job/main/badge/icon)](https://ci.tlcpack.ai/job/flashinfer-ci/job/main/)
[![Documentation](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml/badge.svg)](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml)

## FlashInfer for Windows

FlashInfer Windows build & kernels. This repository will be updated when new versions of FlashInfer are released.

**Don't open a new Issue to request a specific commit build. Wait for a new stable release.**

**Don't open Issues for general FlashInfer questions or non Windows related problems. Only Windows specific issues.** Any Issue opened that is not Windows specific will be closed automatically.

**Don't request a wheel for your specific environment.** Currently, the only wheels I will publish are for Python 3.12 + CUDA 12.4 + torch 2.6.0. If you have another versions, build your own wheel from source by following the instructions below.

### Windows instructions:

#### Installing an existing release wheel:

1. Ensure that you have the correct Python, CUDA and Torch version of the wheel. The Python, CUDA and Torch versions of the wheel are specified in the release version.
2. Download the wheel from the release version of your preference.
3. Install it with ```pip install DOWNLOADED_WHEEL_PATH```

#### Building from source:

##### Pre-requisites

A Visual Studio 2019 or newer is required to launch the compiler x64 environment. The installation path is referred in the instructions as VISUAL_STUDIO_INSTALL_PATH. For example, for Visual Studio 2022 default installation, replace VISUAL_STUDIO_INSTALL_PATH with C:\Program Files\Microsoft Visual Studio\2022\Community

CUDA path will be found automatically if you have the bin folder in your PATH, or have the CUDA installation path settled on well-known environment vars like CUDA_ROOT, CUDA_HOME or CUDA_PATH.

If none of these are present, make sure to set the environment variable before starting the build:
set CUDA_ROOT=CUDA_INSTALLATION_PATH

##### Instructions

1. Open a Command Line (cmd.exe)
2. Execute ```VISUAL_STUDIO_INSTALL_PATH\VC\Auxiliary\Build\vcvarsall.bat x64```
3. Clone the FlashInfer repository: ```cd C:\ & git clone --recurse-submodules https://github.com/SystemPanic/flashinfer-windows.git```
4. Change the working directory to the cloned repository path, for example: ```cd C:\flashinfer-windows```
5. Set the following environment variables:
```
set DISTUTILS_USE_SDK=1
#(replace 10 with your desired cpu threads to use in parallel to speed up compilation)
set MAX_JOBS=10

#(Optional) To build only against your specific GPU CUDA arch (to speed up compilation),
#replace YOUR_CUDA_ARCH with your CUDA arch number. For example, for RTX 4090: set TORCH_CUDA_ARCH_LIST=8.9
set TORCH_CUDA_ARCH_LIST=YOUR_CUDA_ARCH
set FLASHINFER_CUDA_ARCH_LIST=YOUR_CUDA_ARCH
```
6. Build & install:

Make sure to install tvm_ffi with pip, then go to pip site-packages/tvm_ffi/include/tvm/ffi/container/tensor.h and add `class Tensor;` after the first tvm ffi namespaces declaration (L41).

```
#For JIT wheel:
python -m build --no-isolation --wheel
#Replace FLASHINFERVERSION with the corresponding flashinfer version, for example: 0.2.6.post1
pip install dist\flashinfer_python-FLASHINFERVERSION-py3-none-any.whl

#For jit cache (AOT) wheel:
cd flashinfer-jit-cache
python -m build --no-isolation --wheel
#Replace FLASHINFERVERSION with the corresponding flashinfer version, for example: 0.2.6.post1
pip install flashinfer-jit-cache\dist\flashinfer_python-FLASHINFERVERSION-cp39-abi3-win_amd64.whl

#For Cubin wheel:
cd flashinfer-cubin
python -m build --no-isolation --wheel
#Replace FLASHINFERVERSION with the corresponding flashinfer version, for example: 0.2.6.post1
pip install flashinfer-cubin\dist\flashinfer_cubin-FLASHINFERVERSION-py3-none-any.whl

```
7. Build folder cleaning: Due to 260 chars path constraints on Windows, a custom build folder is generated at `C:\_fib` by default. To clean the custom build folder after wheel generation, remove the folder manually.

---

**FlashInfer** is a library and kernel generator for inference that delivers state-of-the-art performance across diverse GPU architectures. It provides unified APIs for attention, GEMM, and MoE operations with multiple backend implementations including FlashAttention-2/3, cuDNN, CUTLASS, and TensorRT-LLM.

## Why FlashInfer?

- **State-of-the-art Performance**: Optimized kernels for prefill, decode, and mixed batching scenarios
- **Multiple Backends**: Automatically selects the best backend for your hardware and workload
- **Modern Architecture Support**: Support for SM75 (Turing) and later (through Blackwell)
- **Low-Precision Compute**: FP8 and FP4 quantization for attention, GEMM, and MoE operations
- **Production-Ready**: CUDAGraph and torch.compile compatible for low-latency serving

## Core Features

### Attention Kernels
- **Paged and Ragged KV-Cache**: Efficient memory management for dynamic batch serving
- **Decode, Prefill, and Append**: Optimized kernels for all attention phases
- **MLA Attention**: Native support for DeepSeek's Multi-Latent Attention
- **Cascade Attention**: Memory-efficient hierarchical KV-Cache for shared prefixes
- **Sparse Attention**: Block-sparse and variable block-sparse patterns
- **POD-Attention**: Fused prefill+decode for mixed batching

### GEMM & Linear Operations
- **FP8 GEMM**: Per-tensor and groupwise scaling
- **FP4 GEMM**: NVFP4 and MXFP4 matrix multiplication for Blackwell GPUs
- **Grouped GEMM**: Efficient batched matrix operations for LoRA and multi-expert routing

### Mixture of Experts (MoE)
- **Fused MoE Kernels**
- **Multiple Routing Methods**: DeepSeek-V3, Llama-4, and standard top-k routing
- **Quantized MoE**: FP8 and FP4 expert weights with block-wise scaling

### Sampling & Decoding
- **Sorting-Free Sampling**: Efficient Top-K, Top-P, and Min-P without sorting
- **Speculative Decoding**: Chain speculative sampling support

### Communication
- **AllReduce**: Custom implementations
- **Multi-Node NVLink**: MNNVL support for multi-node inference
- **NVSHMEM Integration**: For distributed memory operations

### Other Operators
- **RoPE**: LLaMA-style rotary position embeddings (including LLaMA 3.1)
- **Normalization**: RMSNorm, LayerNorm, Gemma-style fused operations
- **Activations**: SiLU, GELU with fused gating

## GPU Support

| Architecture | Compute Capability | Example GPUs |
|--------------|-------------------|------|
| Turing | SM 7.5 | T4, RTX 20 series |
| Ampere | SM 8.0, 8.6 | A100, A10, RTX 30 series |
| Ada Lovelace | SM 8.9 | L4, L40, RTX 40 series |
| Hopper | SM 9.0 | H100, H200 |
| Blackwell | SM 10.0, 10.3 | B200, B300 |
| Blackwell | SM 12.0, 12.1 | RTX 50 series, DGX Spark, Jetson Thor |

## News

Latest: [![GitHub Release](https://img.shields.io/github/v/release/flashinfer-ai/flashinfer)](https://github.com/flashinfer-ai/flashinfer/releases/latest)

Notable updates:
- [2025-10-08] Blackwell support added in [v0.4.0](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.4.0)
- [2025-03-10] [Blog Post](https://flashinfer.ai/2025/03/10/sampling.html) Sorting-Free GPU Kernels for LLM Sampling, which explains the design of sampling kernels in FlashInfer.

## Getting Started

### Installation

**Quickstart:**

```bash
pip install flashinfer-python
```

**Package Options:**

- **flashinfer-python**: Core package that compiles/downloads kernels on first use
- **flashinfer-cubin**: Pre-compiled kernel binaries for all supported GPU architectures
- **flashinfer-jit-cache**: Pre-built kernel cache for specific CUDA versions

**For faster initialization and offline usage**, install the optional packages to have most kernels pre-compiled:

```bash
pip install flashinfer-python flashinfer-cubin
# JIT cache (replace cu129 with your CUDA version)
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
```

### Verify Installation

```bash
flashinfer show-config
```

### Basic Usage

```python
import torch
import flashinfer

# Single decode attention
q = torch.randn(32, 128, device="cuda", dtype=torch.float16)  # [num_qo_heads, head_dim]
k = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)  # [kv_len, num_kv_heads, head_dim]
v = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)

output = flashinfer.single_decode_with_kv_cache(q, k, v)
```

See [documentation](https://docs.flashinfer.ai/) for comprehensive API reference and tutorials.

### Install from Source

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
python -m pip install -v .
```

**For development**, install in editable mode:

```bash
python -m pip install --no-build-isolation -e . -v
```

Build optional packages:

```bash
# flashinfer-cubin
cd flashinfer-cubin
python -m build --no-isolation --wheel
python -m pip install dist/*.whl
```

```bash
# flashinfer-jit-cache (customize for your target GPUs)
export FLASHINFER_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 10.3a 11.0a 12.0f"
cd flashinfer-jit-cache
python -m build --no-isolation --wheel
python -m pip install dist/*.whl
```

For more details, see the [Install from Source documentation](https://docs.flashinfer.ai/installation.html#install-from-source).

### Nightly Builds

```bash
pip install -U --pre flashinfer-python --index-url https://flashinfer.ai/whl/nightly/ --no-deps
pip install flashinfer-python  # Install dependencies from PyPI
pip install -U --pre flashinfer-cubin --index-url https://flashinfer.ai/whl/nightly/
# JIT cache (replace cu129 with your CUDA version)
pip install -U --pre flashinfer-jit-cache --index-url https://flashinfer.ai/whl/nightly/cu129
```

### CLI Tools

FlashInfer provides several CLI commands for configuration, module management, and development:

```bash
# Verify installation and view configuration
flashinfer show-config

# List and inspect modules
flashinfer list-modules
flashinfer module-status

# Manage artifacts and cache
flashinfer download-cubin
flashinfer clear-cache

# For developers: generate compile_commands.json for IDE integration
flashinfer export-compile-commands [output_path]
```

For complete documentation, see the [CLI reference](https://docs.flashinfer.ai/cli.html).

## API Logging

FlashInfer provides comprehensive API logging for debugging. Enable it using environment variables:

```bash
# Enable logging (levels: 0=off (default), 1=basic, 3=detailed, 5=statistics)
export FLASHINFER_LOGLEVEL=3

# Set log destination (stdout (default), stderr, or file path)
export FLASHINFER_LOGDEST=stdout
```

For detailed information about logging levels, configuration, and advanced features, see [Logging](https://docs.flashinfer.ai/logging.html) in our documentation.

## Custom Attention Variants

Users can customize their own attention variants with additional parameters. For more details, refer to our [JIT examples](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/utils/test_jit_example.py).

## CUDA Support

**Supported CUDA Versions:** 12.6, 12.8, 13.0, 13.1

> **Note:** FlashInfer strives to follow PyTorch's supported CUDA versions plus the latest CUDA release.

## Adoption

FlashInfer powers inference in:

- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference)
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
- [LightLLM](https://github.com/ModelTC/lightllm)
- [lorax](https://github.com/predibase/lorax)
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM)

## Acknowledgement

FlashInfer is inspired by [FlashAttention](https://github.com/dao-AILab/flash-attention/), [vLLM](https://github.com/vllm-project/vllm), [stream-K](https://arxiv.org/abs/2301.03598), [CUTLASS](https://github.com/nvidia/cutlass), and [AITemplate](https://github.com/facebookincubator/AITemplate).

## Citation

If you find FlashInfer helpful in your project or research, please consider citing our [paper](https://arxiv.org/abs/2501.01005):

```bibtex
@article{ye2025flashinfer,
    title = {FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving},
    author = {
      Ye, Zihao and
      Chen, Lequn and
      Lai, Ruihang and
      Lin, Wuwei and
      Zhang, Yineng and
      Wang, Stephanie and
      Chen, Tianqi and
      Kasikci, Baris and
      Grover, Vinod and
      Krishnamurthy, Arvind and
      Ceze, Luis
    },
    journal = {arXiv preprint arXiv:2501.01005},
    year = {2025},
    url = {https://arxiv.org/abs/2501.01005}
}
```