<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-black-background.png?raw=true">
    <img alt="FlashInfer" src="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-white-background.png?raw=true" width=55%>
  </picture>
</p>
<h1 align="center">
Kernel Library for LLM Serving
</h1>

<p align="center">
| <a href="https://flashinfer.ai"><b>Blog</b></a> | <a href="https://docs.flashinfer.ai"><b>Documentation</b></a> | <a href="https://join.slack.com/t/flashinfer/shared_invite/zt-379wct3hc-D5jR~1ZKQcU00WHsXhgvtA"><b>Slack</b></a> |  <a href="https://github.com/orgs/flashinfer-ai/discussions"><b>Discussion Forum</b></a> |
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
```
6. Build & install:
```
#For AOT wheel:
python -m flashinfer.aot
python -m build --no-isolation --wheel
#Replace FLASHINFERVERSION with the corresponding flashinfer version, for example: 0.2.6.post1
pip install dist\flashinfer_python-FLASHINFERVERSION-cp39-abi3-win_amd64.whl

#For JIT wheel:
python setup.py bdist_wheel --jit
#Replace FLASHINFERVERSION with the corresponding flashinfer version, for example: 0.2.6.post1
pip install dist\flashinfer_python-FLASHINFERVERSION-py3-none-any.whl
```
7. Build folder cleaning: Due to 260 chars path constraints on Windows, a custom build folder is generated at `C:\_fib` by default. To clean the custom build folder after wheel generation, remove the folder manually or use `python setup.py clean`.

---

FlashInfer is a library and kernel generator for Large Language Models that provides high-performance implementation of LLM GPU kernels such as FlashAttention, SparseAttention, PageAttention, Sampling, and more. FlashInfer focuses on LLM serving and inference, and delivers state-of-the-art performance across diverse scenarios.

Check our [v0.2 release blog](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html) for new features!

The core features of FlashInfer include:
1. **Efficient Sparse/Dense Attention Kernels**: Efficient single/batch attention for sparse(paged)/dense KV-storage on CUDA Cores and Tensor Cores (both FA2 & FA3) templates. The vector-sparse attention can achieve 90% of the bandwidth of dense kernels with same problem size.
2. **Load-Balanced Scheduling**: FlashInfer decouples `plan`/`run` stage of attention computation where we schedule the computation of variable-length inputs in `plan` stage to alleviate load-imbalance issue.
3. **Memory Efficiency**: FlashInfer offers [Cascade Attention](https://docs.flashinfer.ai/api/cascade.html#flashinfer.cascade.MultiLevelCascadeAttentionWrapper) for hierarchical KV-Cache, and implements Head-Query fusion for accelerating Grouped-Query Attention, and efficient kernels for low-precision attention and fused-RoPE attention for compressed KV-Cache.
4. **Customizable Attention**: Bring your own attention variants through JIT-compilation.
5. **CUDAGraph and torch.compile Compatibility**: FlashInfer kernels can be captured by CUDAGraphs and torch.compile for low-latency inference.
6. **Efficient LLM-specific Operators**: High-Performance [fused kernel for Top-P, Top-K/Min-P sampling](https://docs.flashinfer.ai/api/sampling.html) without the need to sorting.

FlashInfer supports PyTorch, TVM and C++ (header-only) APIs, and can be easily integrated into existing projects.

## News
- [Mar 10, 2025] [Blog Post](https://flashinfer.ai/2025/03/10/sampling.html) Sorting-Free GPU Kernels for LLM Sampling, which explains the design of sampling kernels in FlashInfer.
- [Mar 1, 2025] Checkout flashinfer's [intra-kernel profiler](https://github.com/flashinfer-ai/flashinfer/tree/main/profiler) for visualizing the timeline of each threadblock in GPU kernels.
- [Dec 16, 2024] [Blog Post](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html) FlashInfer 0.2 - Efficient and Customizable Kernels for LLM Inference Serving
- [Sept 2024] We've launched a [Slack](https://join.slack.com/t/flashinfer/shared_invite/zt-2r93kj2aq-wZnC2n_Z2~mf73N5qnVGGA) workspace for Flashinfer users and developers. Join us for timely support, discussions, updates and knowledge sharing!
- [Jan 31, 2024] [Blog Post](https://flashinfer.ai/2024/01/08/cascade-inference.html) Cascade Inference: Memory-Efficient Shared Prefix Batch Decoding
- [Jan 31, 2024] [Blog Post](https://flashinfer.ai/2024/01/03/introduce-flashinfer.html) Accelerating Self-Attentions for LLM Serving with FlashInfer

## Getting Started

Using our PyTorch API is the easiest way to get started:

### Install from PyPI

FlashInfer is available as a Python package for Linux. Install the core package with:

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
# JIT cache package (replace cu129 with your CUDA version: cu128, cu129, or cu130)
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
```

This eliminates compilation and downloading overhead at runtime.

### Install from Source

Build the core package from source:

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
python -m pip install -v .
```

**For development**, install in editable mode:
```bash
python -m pip install --no-build-isolation -e . -v
```

**Build optional packages:**

`flashinfer-cubin`:
```bash
cd flashinfer-cubin
python -m build --no-isolation --wheel
python -m pip install dist/*.whl
```

`flashinfer-jit-cache` (customize `FLASHINFER_CUDA_ARCH_LIST` for your target GPUs):
```bash
export FLASHINFER_CUDA_ARCH_LIST="7.5 8.0 8.9 10.0a 10.3a 12.0a"
cd flashinfer-jit-cache
python -m build --no-isolation --wheel
python -m pip install dist/*.whl
```

For more details, see the [Install from Source documentation](https://docs.flashinfer.ai/installation.html#install-from-source).

### Install Nightly Build

Nightly builds are available for testing the latest features:

```bash
# Core and cubin packages
pip install -U --pre flashinfer-python --index-url https://flashinfer.ai/whl/nightly/ --no-deps # Install the nightly package from custom index, without installing dependencies
pip install flashinfer-python  # Install flashinfer-python's dependencies from PyPI
pip install -U --pre flashinfer-cubin --index-url https://flashinfer.ai/whl/nightly/
# JIT cache package (replace cu129 with your CUDA version: cu128, cu129, or cu130)
pip install -U --pre flashinfer-jit-cache --index-url https://flashinfer.ai/whl/nightly/cu129
```

### Verify Installation

After installation, verify that FlashInfer is correctly installed and configured:

```bash
flashinfer show-config
```

This command displays:
- FlashInfer version and installed packages (flashinfer-python, flashinfer-cubin, flashinfer-jit-cache)
- PyTorch and CUDA version information
- Environment variables and artifact paths
- Downloaded cubin status and module compilation status

### Trying it out

Below is a minimal example of using FlashInfer's single-request decode/append/prefill attention kernels:

```python
import torch
import flashinfer

kv_len = 2048
num_kv_heads = 32
head_dim = 128

k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)

# decode attention

num_qo_heads = 32
q = torch.randn(num_qo_heads, head_dim).half().to(0)

o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ROPE_LLAMA") # decode with LLaMA style RoPE on-the-fly

# append attention
append_qo_len = 128
q = torch.randn(append_qo_len, num_qo_heads, head_dim).half().to(0) # append attention, the last 128 tokens in the KV-Cache are the new tokens
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True) # append attention without RoPE on-the-fly, apply causal mask
o_rope_on_the_fly = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, pos_encoding_mode="ROPE_LLAMA") # append attention with LLaMA style RoPE on-the-fly, apply causal mask

# prefill attention
qo_len = 2048
q = torch.randn(qo_len, num_qo_heads, head_dim).half().to(0) # prefill attention
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False) # prefill attention without RoPE on-the-fly, do not apply causal mask
```

Check out [documentation](https://docs.flashinfer.ai/) for usage of batch decode/append/prefill kernels and shared-prefix cascading kernels.

## Custom Attention Variants

Starting from FlashInfer v0.2, users can customize their own attention variants with additional parameters. For more details, refer to our [JIT examples](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/utils/test_jit_example.py).

## GPU Support

FlashInfer currently provides support for NVIDIA SM architectures 75 and higher and beta support for 103, 110, 120, and 121.

## Adoption

We are thrilled to share that FlashInfer is being adopted by many cutting-edge projects, including but not limited to:
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
- [Punica](https://github.com/punica-ai/punica)
- [SGLang](https://github.com/sgl-project/sglang)
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM)
- [vLLM](https://github.com/vllm-project/vllm)
- [TGI](https://github.com/huggingface/text-generation-inference)
- [lorax](https://github.com/predibase/lorax)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [LightLLM](https://github.com/ModelTC/lightllm)

## Acknowledgement

FlashInfer is inspired by [FlashAttention 1&2](https://github.com/dao-AILab/flash-attention/), [vLLM](https://github.com/vllm-project/vllm), [stream-K](https://arxiv.org/abs/2301.03598), [cutlass](https://github.com/nvidia/cutlass) and [AITemplate](https://github.com/facebookincubator/AITemplate) projects.

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
