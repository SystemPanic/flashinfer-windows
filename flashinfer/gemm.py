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

import functools
from types import SimpleNamespace
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm90a_nvcc_flags, sm100a_nvcc_flags
from .utils import (
    _get_cache_buf,
    determine_gemm_backend,
    get_indptr,
    is_float8,
    register_custom_op,
    register_fake_op,
)


def gen_gemm_module() -> JitSpec:
    return gen_jit_spec(
        "gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "bmm_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_gemm_ops.cu",
        ],
        extra_ldflags=["-lcublas", "-lcublasLt"],
    )


@functools.cache
def get_gemm_module():
    module = gen_gemm_module().build_and_load()

    # torch library for bmm_fp8

    @register_custom_op("flashinfer::bmm_fp8", mutates_args=("workspace_buffer", "D"))
    def bmm_fp8(
        workspace_buffer: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        D: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> None:
        cublas_handle = torch.cuda.current_blas_handle()
        module.bmm_fp8.default(
            A,
            B,
            D,
            A_scale,
            B_scale,
            workspace_buffer,
            cublas_handle,
        )

    @register_fake_op("flashinfer::bmm_fp8")
    def _fake_bmm_fp8(
        workspace_buffer: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        D: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> None:
        pass

    # torch library for cutlass_segment_gemm

    @register_custom_op("flashinfer::cutlass_segment_gemm", mutates_args=("y"))
    def cutlass_segment_gemm(
        workspace_buffer: torch.Tensor,
        all_problems: torch.Tensor,
        x_data: torch.Tensor,
        w_data: torch.Tensor,
        y_data: torch.Tensor,
        x_ld: torch.Tensor,
        w_ld: torch.Tensor,
        y_ld: torch.Tensor,
        y: torch.Tensor,
        empty_x_data: torch.Tensor,
        weight_column_major: bool,
    ) -> None:
        module.cutlass_segment_gemm.default(
            workspace_buffer,
            all_problems,
            x_data,
            w_data,
            y_data,
            x_ld,
            w_ld,
            y_ld,
            empty_x_data,
            weight_column_major,
        )

    @register_fake_op("flashinfer::cutlass_segment_gemm")
    def _fake_cutlass_segment_gemm(
        workspace_buffer: torch.Tensor,
        all_problems: torch.Tensor,
        x_data: torch.Tensor,
        w_data: torch.Tensor,
        y_data: torch.Tensor,
        x_ld: torch.Tensor,
        w_ld: torch.Tensor,
        y_ld: torch.Tensor,
        y: torch.Tensor,
        empty_x_data: torch.Tensor,
        weight_column_major: bool,
    ) -> None:
        pass

    # Register the module
    _gemm_module = SimpleNamespace(
        bmm_fp8=bmm_fp8,
        cutlass_segment_gemm=cutlass_segment_gemm,
    )

    return _gemm_module


def gen_gemm_sm100_module() -> JitSpec:
    return gen_jit_spec(
        "gemm_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR / "gemm_groupwise_sm100.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_fp8_groupwise_sm100.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_mxfp4_groupwise_sm100.cu",
            jit_env.FLASHINFER_CSRC_DIR / "gemm_sm100_pybind.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_sm100_pybind.cu",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags,
    )


@functools.cache
def get_gemm_sm100_module():
    module = gen_gemm_sm100_module().build_and_load()

    return module


def gen_gemm_sm90_module() -> JitSpec:
    return gen_jit_spec(
        "gemm_sm90",
        [
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_gemm_sm90_ops.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_f16_f16_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_bf16_bf16_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_e4m3_f16_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_e5m2_f16_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_e4m3_bf16_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm_e5m2_bf16_sm90.cu",
        ],
        extra_cuda_cflags=sm90a_nvcc_flags,
    )


@functools.cache
def get_gemm_sm90_module():
    module = gen_gemm_sm90_module().build_and_load()

    # torch library for cutlass_segment_gemm_sm90

    @register_custom_op(
        "flashinfer::cutlass_segment_gemm_sm90",
        mutates_args=("workspace_buffer", "y"),
    )
    def cutlass_segment_gemm_sm90(
        workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        all_problems: torch.Tensor,
        x_data: torch.Tensor,
        w_data: torch.Tensor,
        y_data: torch.Tensor,
        x_stride: torch.Tensor,
        w_stride: torch.Tensor,
        y_stride: torch.Tensor,
        y: torch.Tensor,
        empty_x_data: torch.Tensor,
        empty_y_data: torch.Tensor,
        weight_column_major: bool,
    ) -> None:
        module.cutlass_segment_gemm_sm90.default(
            workspace_buffer,
            int_workspace_buffer,
            all_problems,
            x_data,
            w_data,
            y_data,
            x_stride,
            w_stride,
            y_stride,
            empty_x_data,
            empty_y_data,
            weight_column_major,
        )

    @register_fake_op("flashinfer::cutlass_segment_gemm_sm90")
    def _fake_cutlass_segment_gemm_sm90(
        workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        all_problems: torch.Tensor,
        x_data: torch.Tensor,
        w_data: torch.Tensor,
        y_data: torch.Tensor,
        x_stride: torch.Tensor,
        w_stride: torch.Tensor,
        y_stride: torch.Tensor,
        y: torch.Tensor,
        empty_x_data: torch.Tensor,
        empty_y_data: torch.Tensor,
        weight_column_major: bool,
    ) -> None:
        pass

    # Register the module
    return SimpleNamespace(
        cutlass_segment_gemm_sm90=cutlass_segment_gemm_sm90,
    )


def launch_compute_sm80_group_gemm_args(
    x: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    w_column_major: bool,
    batch_size: int,
    seg_indptr: torch.Tensor,
    weight_indices: Optional[torch.Tensor] = None,
):
    device = x.device
    prob_type = torch.int32  # problem sizes -> int
    ptr_type = torch.int64  # pointers -> int64_t
    ld_type = torch.int64  # strides -> int64_t

    seg_indptr = seg_indptr.to(ptr_type)
    if weight_indices is not None:
        weight_indices = weight_indices.to(ptr_type)

    d_out = weights.size(1) if w_column_major else weights.size(2)
    d_in = weights.size(2) if w_column_major else weights.size(1)

    all_problems = torch.empty((batch_size, 3), dtype=prob_type, device=device)

    x_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    w_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    y_data = torch.empty(batch_size, dtype=ptr_type, device=device)

    x_stride_data = torch.empty(batch_size, dtype=ld_type, device=device)
    w_stride_data = torch.empty(batch_size, dtype=ld_type, device=device)
    y_stride_data = torch.empty(batch_size, dtype=ld_type, device=device)

    from .triton.gemm import compute_sm80_group_gemm_args

    compute_sm80_group_gemm_args[(batch_size,)](
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
        x,
        weights,
        y,
        seg_indptr,
        weight_indices,
        d_in,
        d_out,
        w_column_major,
    )

    return (
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
    )


def launch_compute_sm90_group_gemm_args(
    x: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    w_column_major: bool,
    batch_size: int,
    seg_indptr: torch.Tensor,
    weight_indices: Optional[torch.Tensor] = None,
):
    device = x.device
    prob_type = torch.int32  # problem sizes -> int
    ptr_type = torch.int64  # pointers -> int64_t
    stride_type = torch.int64  # strides -> int64_t

    seg_indptr = seg_indptr.to(ptr_type)
    if weight_indices is not None:
        weight_indices = weight_indices.to(ptr_type)

    d_out = weights.size(1) if w_column_major else weights.size(2)
    d_in = weights.size(2) if w_column_major else weights.size(1)

    all_problems = torch.empty((batch_size, 3), dtype=prob_type, device=device)

    x_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    w_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    y_data = torch.empty(batch_size, dtype=ptr_type, device=device)

    x_stride_data = torch.empty(batch_size, dtype=stride_type, device=device)
    w_stride_data = torch.empty(batch_size, dtype=stride_type, device=device)
    y_stride_data = torch.empty(batch_size, dtype=stride_type, device=device)

    from .triton.gemm import compute_sm90_group_gemm_args

    compute_sm90_group_gemm_args[(batch_size,)](
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
        x,
        weights,
        y,
        seg_indptr,
        weight_indices,
        d_in,
        d_out,
        w_column_major,
    )

    return (
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
    )


class SegmentGEMMWrapper:
    r"""Wrapper for segment GEMM kernels.

    Example
    -------
    >>> import torch
    >>> from flashinfer import SegmentGEMMWrapper
    >>> # create a 1MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    >>> segment_gemm = SegmentGEMMWrapper(workspace_buffer)
    >>> seq_lens = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    >>> # create packed input tensor (10 = 1 + 2 + 3 + 4)
    >>> x = torch.randn(10, 128, device="cuda", dtype=torch.float16)
    >>> # create weight tensor with 4 weights, each with 128 input and 256 output channels, column major
    >>> weights = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16)
    >>> # compute the segment GEMM
    >>> y = segment_gemm.run(x, weights, 4, True, seg_lens=seq_lens)
    >>> y.shape
    torch.Size([10, 256])
    >>> y_ref_0 = torch.matmul(x[:1], weights[0].t())
    >>> torch.allclose(y[:1], y_ref_0)
    True
    >>> y_ref_1 = torch.matmul(x[1:3], weights[1].t())
    >>> torch.allclose(y[1:3], y_ref_1)
    True
    >>> y_ref_2 = torch.matmul(x[3:6], weights[2].t())
    >>> torch.allclose(y[3:6], y_ref_2)
    True
    >>> y_ref_3 = torch.matmul(x[6:], weights[3].t())
    >>> torch.allclose(y[6:], y_ref_3)
    True
    >>>
    >>> # another example with weight indices
    >>> weight_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda")
    >>> y = segment_gemm.run(x, weights, 4, True, seg_lens=seq_lens, weight_indices=weight_indices)
    >>> y.shape
    torch.Size([10, 256])
    >>> y_ref_0 = torch.matmul(x[:1], weights[0].t())
    >>> torch.allclose(y[:1], y_ref_0)
    True
    >>> y_ref_1 = torch.matmul(x[1:3], weights[1].t())
    >>> torch.allclose(y[1:3], y_ref_1)
    True
    >>> y_ref_2 = torch.matmul(x[3:6], weights[0].t())
    >>> torch.allclose(y[3:6], y_ref_2)
    True
    >>> y_ref_3 = torch.matmul(x[6:], weights[1].t())
    >>> torch.allclose(y[6:], y_ref_3)
    True
    """

    def __init__(
        self, float_workspace_buffer: torch.Tensor, backend: str = "auto"
    ) -> None:
        r"""Initialize the wrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The workspace buffer for the kernels, we use it for storing intermediate results in cutlass
            segment GEMM kernels. Encouraged size is 128MB.
        """
        self._int_workspace_buffer = torch.empty(
            (1024 * 1024,), dtype=torch.int8, device=float_workspace_buffer.device
        )
        self._float_workspace_buffer = float_workspace_buffer
        self.backend = backend

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer for the kernels.
        int_workspace_buffer : torch.Tensor
            The new int workspace buffer for the kernels.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer

    def run(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        out: Optional[torch.Tensor] = None,
        seg_lens: Optional[torch.Tensor] = None,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Run the segment GEMM kernel.

        Compute the matrix multiplication between a batch of input tensor (with variable number of rows, but fixed
        number of columns) and a batch of weight tensor with fixed number of rows and columns:

        .. math::

            y[i] = x[i] \times W[i]

        if :attr:`weight_indices` is provided, we will select the weight tensor based on the indices in the
        :attr:`weight_indices` tensor:

        .. math::

            y[i] = x[i] \times W[\text{weight_indices}[i]]

        We use Ragged Tensor to represent the input tensor :attr:`x` and the output tensor :attr:`y`, and each x[i]
        is a segment of the concatenated tensor. Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details.
        We use a ``seg_len`` or ``seg_indptr`` tensor (either would work) to indicate the start and end of each segment,
        where the ``seg_indptr`` is the cumulative sum of the ``seg_lens`` tensor (with an additional 0 at the beginning):

        .. math::

            \text{seg_indptr}[i] = \sum_{j=0}^{i-1} \text{seg_lens}[j], \quad \text{seg_indptr}[0] = 0

        - If ``seg_lens`` is provided, then :attr:`x` has shape ``(sum(seg_lens), d_in)`` and :attr:`y` has shape
            ``(sum(seg_lens), d_out)``, where ``d_in`` is the number of columns of the input tensor and ``d_out`` is the
            number of columns of the output tensor.
        - If ``seg_indptr`` is provided, then :attr:`x` has shape ``(seg_indptr[-1], d_in)`` and :attr:`y` has shape
            ``(seg_indptr[-1], d_out)``.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape ``(sum(seg_lens), d_in)``.
        weights : torch.Tensor
            The 3D weight tensor with shape ``(num_weights, d_in, d_out)`` if :attr:`weight_column_major` is ``False``,
            or ``(num_weights, d_out, d_in)`` if :attr:`weight_column_major` is ``True``.
        batch_size : int
            The number of segments.
        weight_column_major : bool
            Whether the weight tensor is column major.
        out : Optional[torch.Tensor]
            The output tensor, with shape ``(sum(seg_lens), d_out)``.
            If not provided, a new tensor will be created internally.
        seg_lens : Optional[torch.Tensor]
            The length of each segment, with shape ``(batch_size,)``, expects a 1D tensor of dtype ``torch.int64``.
        seg_indptr : Optional[torch.Tensor]
            The indptr of the segments, with shape ``(batch_size + 1,)``, expects a 1D tensor of dtype ``torch.int64``.
            If this is provided, then :attr:`seg_lens` will be ignored, otherwise ``seg_indptr`` will be computed
            internally from :attr:`seg_lens`.
        weight_indices : Optional[torch.Tensor]
            The indices of the weight tensor to be selected for each segment, with shape ``(batch_size,)``.
            Expects a 1D tensor of dtype ``torch.int64``.
            If this is provided, then the weight tensor will be selected based on the indices in this tensor.

        Returns
        -------
        torch.Tensor
            The output tensor with shape ``(sum(seg_lens), d_out)``.
        """
        if seg_lens is None and seg_indptr is None:
            raise ValueError("Either seg_lens or seg_indptr should be provided.")
        if seg_indptr is None:
            seg_indptr = get_indptr(seg_lens.to(x))
        if weight_indices is None:
            # create an empty CPU tensor as placeholder
            weight_indices = torch.empty(0, dtype=torch.int64)
        cumulative_batch_size = x.size(0)
        d_out = weights.size(1) if weight_column_major else weights.size(2)
        if out is None:
            if is_float8(x):
                out_dtype = torch.bfloat16
            else:
                out_dtype = x.dtype
            out = torch.zeros(
                (cumulative_batch_size, d_out), dtype=out_dtype, device=x.device
            )
        else:
            if out.shape != (cumulative_batch_size, d_out):
                raise ValueError(
                    f"Output tensor shape mismatch, expected {cumulative_batch_size, d_out}, got {out.shape}"
                )
        empty_x_data = torch.empty(0, dtype=x.dtype, device=x.device)
        empty_y_data = torch.empty(0, dtype=out.dtype, device=out.device)

        if self.backend == "auto":
            backend = determine_gemm_backend(x.device)
        else:
            backend = self.backend

        if backend == "sm90":
            (
                all_problems,
                x_data,
                w_data,
                y_data,
                x_stride_data,
                w_stride_data,
                y_stride_data,
            ) = launch_compute_sm90_group_gemm_args(
                x,
                weights,
                out,
                weight_column_major,
                batch_size,
                seg_indptr,
                weight_indices,
            )
            get_gemm_sm90_module().cutlass_segment_gemm_sm90(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                all_problems,
                x_data,
                w_data,
                y_data,
                x_stride_data,
                w_stride_data,
                y_stride_data,
                out,  # for torch compile mutates_args
                empty_x_data,  # for kernel type dispatch
                empty_y_data,
                weight_column_major,
            )
        elif backend == "sm80":
            (
                all_problems,
                x_data,
                w_data,
                y_data,
                x_ld_data,
                w_ld_data,
                y_ld_data,
            ) = launch_compute_sm80_group_gemm_args(
                x,
                weights,
                out,
                weight_column_major,
                batch_size,
                seg_indptr,
                weight_indices,
            )
            get_gemm_module().cutlass_segment_gemm(
                self._int_workspace_buffer,
                all_problems,
                x_data,
                w_data,
                y_data,
                x_ld_data,
                w_ld_data,
                y_ld_data,
                out,
                empty_x_data,
                weight_column_major,
            )
        else:
            raise ValueError(f"Unsupported gemm backend: {backend}")
        return out

    forward = run


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""BMM FP8

    Parameters
    ----------
    A: torch.Tensor
        Input tensor, shape (b, m, k), fp8 e4m3 or fp8 e5m2.

    B: torch.Tensor
        Mat2 tensor, shape (b, k, n), should be column major, fp8 e4m3 or fp8 e5m2.

    A_scale: torch.Tensor
        Scale tensor for A, float.

    B_scale: torch.Tensor
        Scale tensor for B, float.

    dtype: torch.dtype
        out dtype, bf16 or fp16.

    out: Optional[torch.Tensor]
        Out tensor, shape (b, m, n), bf16 or fp16, defaults to ``None``.

    Returns
    -------
    out: torch.Tensor
        Out tensor, shape (b, m, n), bf16 or fp16.

    Examples
    --------
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import flashinfer
    >>> def to_float8(x, dtype=torch.float8_e4m3fn):
    ...     finfo = torch.finfo(dtype)
    ...     min_val, max_val = x.aminmax()
    ...     amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    ...     scale = finfo.max / amax
    ...     x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    ...     return x_scl_sat.to(dtype), scale.float().reciprocal()
    >>>
    >>> input = torch.randn([16, 48, 64], device="cuda", dtype=torch.bfloat16)
    >>> input_fp8, input_inv_s = to_float8(input, dtype=torch.float8_e4m3fn)
    >>> # column major weight
    >>> weight = torch.randn([16, 80, 64], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    >>> weight_fp8, weight_inv_s = to_float8(weight, dtype=torch.float8_e4m3fn)
    >>> out = flashinfer.bmm_fp8(input_fp8, weight_fp8, input_inv_s, weight_inv_s, torch.bfloat16)
    >>> out.shape
    torch.Size([16, 48, 80])
    >>> out.dtype
    torch.bfloat16
    """
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )
    workspace_buffer = _get_cache_buf("bmm_fp8_workspace", 32 * 1024 * 1024, A.device)
    get_gemm_module().bmm_fp8(workspace_buffer, A, B, out, A_scale, B_scale)
    return out


def gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Literal["MN", "K"] = "MN",
    mma_sm: int = 1,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Performs matrix multiplication with FP8 data types using groupwise scaling.

    This function implements a GEMM operation that allows for fine-grained control over
    scale granularity across different dimensions. Currently only supported on NVIDIA
    Blackwell architecture.

    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor shape (m, k), fp8 e4m3 or fp8 e5m2.

    b: torch.Tensor
        Column-major input tensor shape (n, k), fp8 e4m3 or fp8 e5m2.

    a_scale: torch.Tensor
        Column-major scale tensor for a, shape ``(m, k // block_size)`` if scale_major_mode is ``K``
        or shape ``(k // block_size, m)`` if scale_major_mode is ``MN``

    b_scale: torch.Tensor
        Row-major scale tensor for b, shape ``(n // block_size, k // block_size)`` if scale_major_k is ``K``
        or shape ``(k // block_size, n // block_size)`` if scale_major_mode is ``MN``

    scale_granularity_mnk: Tuple[int, int, int]
        The granularity of the scale tensor, (m_granularity, n_granularity, k_granularity).

    scale_major_mode: Literal["MN", "K"]
        The layout mode of scale tensor, `MN` for MN-major scale with shape of
        ``(k // block_size, *)`` and `K` for K-major scale with shape of
        ``(*, k // block_size)``

    mma_sm: int
        How many SMs to use for the MMA operation, must be 1 or 2.
        2 is faster when number of rows (M) per group is large (>= 256).

    out: Optional[torch.Tensor]
        Output tensor, shape (m, n). If not specified, we will create an output tensor explicitly.

    out_dtype: Optional[torch.dtype]
        If out is not specified, we will create an output tensor with this dtype.
        Defaults to ``torch.bfloat16``.

    Returns
    -------
    out: torch.Tensor
        Output tensor, shape (m, n).

    Notes
    -----
    The ``m`` should be padded to a multiple of 4 before calling this function, to accommodate the kernel's requirement.
    """
    workspace_buffer = _get_cache_buf(
        "gemm_fp8_nt_groupwise_workspace", 32 * 1024 * 1024, a.device
    )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Shape mismatch. a.shape = {a.shape}, b.shape = {b.shape}")

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Shape mismatch. a.shape[1] = {a.shape[1]}, b.shape[1] = {b.shape[1]}"
        )

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
    else:
        out_dtype = out.dtype

    # NOTE(Zihao): (out_specified, need_padding)
    # (False, False) -> create out_padded tensor explicitly
    # (False, True) -> create out_padded tensor explicitly
    # (True, False) -> use out tensor as out_padded
    # (True, True) -> create out_padded tensor explicitly

    if out is None:
        out = torch.empty(
            a.shape[0],
            b.shape[0],
            device=a.device,
            dtype=out_dtype,
        )

    get_gemm_sm100_module().gemm_fp8_nt_groupwise.default(
        workspace_buffer,
        a,
        b,
        a_scale,
        b_scale,
        out,
        *scale_granularity_mnk,
        scale_major_mode,
        mma_sm,
    )

    return out


def gemm_fp8_nt_blockscaled(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: str = "MN",
    mma_sm: int = 1,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Performs matrix multiplication with FP8 data types using block-scaled scaling.

    Block-scaled scaling is a special case of groupwise scaling where the scale granularity
    is (128, 128, 128).
    """
    return gemm_fp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        scale_granularity_mnk=(128, 128, 128),
        scale_major_mode=scale_major_mode,
        mma_sm=mma_sm,
        out=out,
        out_dtype=out_dtype,
    )


def group_gemm_fp8_nt_groupwise(
    a: torch.Tensor,  # (cum_m, k)
    b: torch.Tensor,  # (batch_size, n, k)
    a_scale: torch.Tensor,  # (k // block_size, cum_m)
    b_scale: torch.Tensor,  # (batch_size, k // block_size, n // block_size)
    m_indptr: torch.Tensor,  # (batch_size + 1, )
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    scale_major_mode: Literal["MN", "K"] = "MN",
    mma_sm: int = 1,
    out: Optional[torch.Tensor] = None,  # (cum_m, n)
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Perform group GEMM with FP8 data types using groupwise scaling. Currently only supported on NVIDIA
    Blackwell architecture.

    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor shape ``(cum_m, k)``, data type is ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
        ``cum_m`` is the cumulative sum of the segment lengths.

    b: torch.Tensor
        Column-major input tensor shape ``(batch_size, n, k)``, data type is ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.

    a_scale: torch.Tensor
        Column-major scale tensor for a, shape ``(cum_m, k // block_size)`` if scale_major_mode is ``K``
        or shape ``(k // block_size, cum_m)`` if scale_major_mode is ``MN``, data type is ``torch.float32``.

    b_scale: torch.Tensor
        Row-major scale tensor for b, shape ``(batch_size, n // block_size, k // block_size)`` if scale_major_mode is ``K``
        shape ``(batch_size, k // block_size, n // block_size)`` if scale_major_mode is ``MN``, data type is ``torch.float32``.

    m_indptr: torch.Tensor
        The indptr of the segment lengths, shape ``(batch_size + 1,)``, data type is ``torch.int32``.
        Element element in ``m_indptr`` must be a multiple of 4.

    scale_granularity_mnk: Tuple[int, int, int]
        The granularity of the scale tensor, (m_granularity, n_granularity, k_granularity).

    scale_major_mode: Literal["MN", "K"]
        The layout mode of scale tensor, `MN` for MN-major scale with shape of
        ``(k // block_size, *)`` and `K` for K-major scale with shape of
        ``(*, k // block_size)``

    mma_sm: int
        How many SMs to use for the MMA operation, must be 1 or 2.
        2 is faster when number of rows (M) per group is large (>= 256).

    out: Optional[torch.Tensor]
        The output tensor, shape ``(cum_m, n)``. If not specified, we will create an output tensor explicitly.

    out_dtype: Optional[torch.dtype]
        The data type of the output tensor, must be ``torch.bfloat16`` or ``torch.float16``.

    Returns
    -------
    out: torch.Tensor
        The output tensor, shape ``(cum_m, n)``.

    Notes
    -----
    Each value in ``m_indptr`` should be padded to a multiple of 4 before calling this function,
    to accommodate the kernel's requirement.
    """
    int_workspace_buffer = _get_cache_buf(
        "group_gemm_fp8_nt_groupwise_int_workspace", 32 * 1024 * 1024, a.device
    )
    float_workspace_buffer = _get_cache_buf(
        "group_gemm_fp8_nt_groupwise_float_workspace", 32 * 1024 * 1024, a.device
    )

    assert a.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    assert b.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    assert a_scale.dtype == torch.float32
    assert b_scale.dtype == torch.float32
    assert m_indptr.dtype == torch.int32
    assert scale_major_mode in ["MN", "K"]
    assert mma_sm in [1, 2]
    if out is None:
        if out_dtype is None:
            out_dtype = torch.bfloat16
    else:
        if out_dtype is None:
            out_dtype = out.dtype
    assert out_dtype in [torch.bfloat16, torch.float16]

    num_groups = m_indptr.shape[0] - 1
    assert b.shape[0] == num_groups
    n = b.shape[1]
    k = b.shape[2]

    # assert a.shape[0] == m_indptr[-1].item()  # Not enabled in consideration of performance
    assert a.shape[1] == k
    align_n = 8
    align_k = 16
    assert n % align_n == 0
    assert k % align_k == 0

    out_shape = (a.shape[0], n)
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=a.device)
    else:
        assert out.shape == out_shape
        assert out.dtype == out_dtype

    get_gemm_sm100_module().group_gemm_fp8_nt_groupwise.default(
        int_workspace_buffer,
        float_workspace_buffer,
        a,
        b,
        a_scale,
        b_scale,
        out,
        m_indptr,
        n,
        k,
        *scale_granularity_mnk,
        scale_major_mode,
        mma_sm,
    )
    return out


def group_gemm_mxfp4_nt_groupwise(
    a: torch.Tensor,  # (cum_m, k)
    b: torch.Tensor,  # (batch_size, n, k // 2)
    a_scale: torch.Tensor,  # (cum_m_padded, k // 32)
    b_scale: torch.Tensor,  # (batch_size, n_padded, k // 32)
    m_indptr: torch.Tensor,  # (batch_size + 1, )
    mma_sm: int = 1,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    swap_ab: bool = True,
    out: Optional[torch.Tensor] = None,  # (cum_m, n)
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Perform group GEMM with MXFP4 data types using groupwise scaling. Currently only supported on NVIDIA
    Blackwell architecture.

    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor, shape ``(cum_m, k)``, data type is ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
        ``cum_m`` is the cumulative sum of the segment lengths.

    b: torch.Tensor
        Column-major input tensor, shape ``(batch_size, n, k // 2)``, data type is ``torch.uint8``.

    a_scale: torch.Tensor
        Column-major scale tensor for a, shape ``(cum_m_padded, k // 32)``, data type is ``torch.uint8``.

    b_scale: torch.Tensor
        Row-major scale tensor for b, shape ``(batch_size, n_padded, k // 32)``, data type is ``torch.uint8``.

    m_indptr: torch.Tensor
        The indptr of the segment lengths, shape ``(batch_size + 1,)``, data type is ``torch.int32``.
        Element element in ``m_indptr`` must be a multiple of 4.

    mma_sm: int
        How many SMs to use for the MMA operation, must be 1 or 2.
        2 is faster when number of rows (M) per group is large (>= 256).

    tile_m: int
        The tile size for the M dimension, must be 128.

    tile_n: int
        The tile size for the N dimension, must be 64, 128, 192, or 256.

    tile_k: int
        The tile size for the K dimension, must be 128 or 256.

    swap_ab: bool
        Whether to swap the A and B tensors.

    out: Optional[torch.Tensor]
        The output tensor, shape ``(cum_m, n)``. If not specified, we will create an output tensor explicitly.

    out_dtype: Optional[torch.dtype]
        The data type of the output tensor, must be ``torch.bfloat16`` or ``torch.float16``.

    Returns
    -------
    out: torch.Tensor
        The output tensor, shape ``(cum_m, n)``.

    Notes
    -----
    Each value in ``m_indptr`` should be padded to a multiple of 4 before calling this function,
    to accommodate the kernel's requirement.
    """
    int_workspace_buffer = _get_cache_buf(
        "group_gemm_mxfp4_nt_groupwise_int_workspace", 32 * 1024 * 1024, a.device
    )
    float_workspace_buffer = _get_cache_buf(
        "group_gemm_mxfp4_nt_groupwise_float_workspace", 32 * 1024 * 1024, a.device
    )

    assert a.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    assert b.dtype == torch.uint8
    assert a_scale.dtype == torch.uint8
    assert b_scale.dtype == torch.uint8
    assert m_indptr.dtype == torch.int32
    assert mma_sm in [1, 2]
    assert tile_m in [128]
    assert tile_n in [64, 128, 192, 256]
    assert tile_k in [128, 256]
    assert swap_ab in [True, False]
    if out is None:
        if out_dtype is None:
            out_dtype = torch.bfloat16
    else:
        if out_dtype is None:
            out_dtype = out.dtype
    assert out_dtype in [torch.bfloat16, torch.float16]

    num_groups = m_indptr.shape[0] - 1
    assert b.shape[0] == num_groups
    n = b.shape[1]
    k = b.shape[2] * 2  # Multiply by 2 because b is e2m1 packed as uint8

    # assert a.shape[0] == m_indptr[-1].item()  # Not enabled in consideration of performance
    assert a.shape[1] == k
    align_n = 8
    align_k = 128
    assert n % align_n == 0
    assert k % align_k == 0

    out_shape = (a.shape[0], n)
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=a.device)
    else:
        assert out.shape == out_shape
        assert out.dtype == out_dtype

    get_gemm_sm100_module().group_gemm_mxfp4_nt_groupwise.default(
        int_workspace_buffer,
        float_workspace_buffer,
        a,
        b,
        a_scale,
        b_scale,
        out,
        m_indptr,
        n,
        k,
        mma_sm,
        tile_m,
        tile_n,
        tile_k,
        swap_ab,
    )
    return out


def pad_indptr_to_multiple_of_4(
    m_indptr: torch.Tensor,
):
    from .triton.gemm import compute_padding_mapping

    batch_size = m_indptr.shape[0] - 1
    m = m_indptr[1:] - m_indptr[:-1]
    m = m + 3 - (m + 3) % 4
    padded_m_indptr = torch.cat((torch.zeros((1,), device=m.device, dtype=m.dtype), m))
    padded_m_indptr = padded_m_indptr.cumsum(dim=0, dtype=padded_m_indptr.dtype)

    m_rank = torch.zeros((m_indptr[-1],), dtype=m_indptr.dtype, device=m_indptr.device)
    padded_m_rank = torch.zeros(
        (m_indptr[-1],), dtype=m_indptr.dtype, device=m_indptr.device
    )

    compute_padding_mapping[(batch_size,)](
        m_indptr, padded_m_indptr, m_rank, padded_m_rank
    )

    return padded_m_indptr, padded_m_rank


def gen_deepgemm_sm100_module() -> SimpleNamespace:
    from flashinfer.deep_gemm import load_all

    load_all()
    return SimpleNamespace(
        group_deepgemm_fp8_nt_groupwise=group_deepgemm_fp8_nt_groupwise,
    )


@functools.cache
def get_deepgemm_sm100_module():
    module = gen_deepgemm_sm100_module()
    return module


def group_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,  # (m, k)
    b: torch.Tensor,  # (batch_size, n, k)
    a_scale: torch.Tensor,  # (m, k // block_size)
    b_scale: torch.Tensor,  # (batch_size, n // block_size, k // block_size)
    m_indices: torch.Tensor,  # (m, )
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,  # (m, n)
    out_dtype: Optional[torch.dtype] = None,
):
    r"""Perform grouped matrix multiplication with FP8 data types using DeepGEMM backend.

    This function performs a grouped GEMM operation where each group in tensor `b` is multiplied
    with the corresponding rows in tensor `a`. The grouping is determined by the `m_indices` tensor,
    which specifies which group each row belongs to. This is particularly useful for scenarios
    like mixture of experts (MoE) where different tokens are routed to different experts.

    The operation can be conceptualized as:
    ```
    for i in range(num_groups):
        row_slice = slice(i * m_per_group, (i + 1) * m_per_group)
        output[row_slice] = a[row_slice] @ b[i].T
    ```

    Currently only supported on NVIDIA Blackwell (SM100) architecture.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor A of shape ``(m, k)`` with FP8 data type (``torch.float8_e4m3fn``).
        This tensor contains all rows that will be multiplied with different groups in `b`.

    b : torch.Tensor
        Input tensor B of shape ``(batch_size, n, k)`` with FP8 data type (``torch.float8_e4m3fn``).
        Each slice ``b[i]`` represents a different group/expert that will be multiplied with
        the corresponding rows in `a`.

    a_scale : torch.Tensor
        Scaling factors for tensor `a` of shape ``(m, k // block_size)`` with ``torch.float32`` dtype.
        These are typically generated from per-token quantization of the original float32 tensor.

    b_scale : torch.Tensor
        Scaling factors for tensor `b` of shape ``(batch_size, n // block_size, k // block_size)``
        with ``torch.float32`` dtype. These are typically generated from per-block quantization
        of the original float32 tensor for each group.

    m_indices : torch.Tensor
        Group assignment tensor of shape ``(m,)`` with ``torch.int32`` dtype. Each element
        specifies which group (index into `b`) the corresponding row in `a` belongs to.
        For example, if ``m_indices[i] = j``, then row ``i`` in `a` will be multiplied with
        group ``j`` in `b`.

    scale_granularity_mnk : Tuple[int, int, int], optional
        The granularity of the scaling factors as ``(m_granularity, n_granularity, k_granularity)``.
        Default is ``(1, 128, 128)`` which means per-token scaling for `a` and 128x128 block
        scaling for `b`.

    out : Optional[torch.Tensor], optional
        Pre-allocated output tensor of shape ``(m, n)``. If not provided, a new tensor will be
        created.

    out_dtype : Optional[torch.dtype], optional
        Data type of the output tensor. If `out` is provided, this parameter is ignored.
        Default is ``torch.bfloat16``.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``(m, n)`` containing the results of the grouped matrix multiplication.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.gemm import group_deepgemm_fp8_nt_groupwise
    >>> from flashinfer.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
    >>>
    >>> # Setup: 2 groups, 128 tokens per group, 4096 hidden size, 2048 expert size
    >>> m_per_group, n, k = 128, 2048, 4096
    >>> group_size = 2
    >>> m = m_per_group * group_size
    >>>
    >>> # Create float32 inputs
    >>> a_f32 = torch.randn(m, k, device="cuda", dtype=torch.float32)
    >>> b_f32 = torch.randn(group_size, n, k, device="cuda", dtype=torch.float32)
    >>>
    >>> # Quantize to FP8 with appropriate scaling
    >>> a_fp8, a_scale = per_token_cast_to_fp8(a_f32)
    >>> b_fp8 = torch.empty_like(b_f32, dtype=torch.float8_e4m3fn)
    >>> b_scale = torch.empty((group_size, n // 128, k // 128), device="cuda", dtype=torch.float32)
    >>> for i in range(group_size):
    ...     b_fp8[i], b_scale[i] = per_block_cast_to_fp8(b_f32[i])
    >>>
    >>> # Create group assignment
    >>> m_indices = torch.empty(m, device="cuda", dtype=torch.int32)
    >>> for i in range(group_size):
    ...     row_slice = slice(i * m_per_group, (i + 1) * m_per_group)
    ...     m_indices[row_slice] = i
    >>>
    >>> # Perform grouped GEMM
    >>> result = group_deepgemm_fp8_nt_groupwise(
    ...     a_fp8, b_fp8, a_scale, b_scale, m_indices, out_dtype=torch.bfloat16
    ... )
    >>> print(result.shape)  # torch.Size([256, 2048])

    Notes
    -----
    - This function requires NVIDIA Blackwell (SM100) architecture
    - The scaling factors should be generated using appropriate quantization functions
      like ``per_token_cast_to_fp8`` for `a` and ``per_block_cast_to_fp8`` for `b`
    - The function internally uses the DeepGEMM backend for optimized FP8 computation
    - All input tensors must be on the same CUDA device
    - The block size for scaling is determined by the ``scale_granularity_mnk`` parameter
    """
    from flashinfer.deep_gemm import m_grouped_fp8_gemm_nt_contiguous

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty(a.shape[0], b.shape[1], dtype=out_dtype, device=a.device)

    m_grouped_fp8_gemm_nt_contiguous(
        (a, a_scale), (b, b_scale), out, m_indices, scale_granularity_mnk
    )

    return out
