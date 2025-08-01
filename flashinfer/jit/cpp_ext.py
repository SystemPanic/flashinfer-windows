# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py

import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.cpp_extension import (
    _TORCH_PATH,
    CUDA_HOME,
    _get_cuda_arch_flags,
    _get_num_workers,
    _get_pybind11_abi_build_flags,
)

from . import env as jit_env


def _get_glibcxx_abi_build_flags() -> List[str]:
    glibcxx_abi_cflags = [
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    ]
    return glibcxx_abi_cflags


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
    needs_device_linking: bool = False,
) -> str:

    system_includes = [
        sysconfig.get_path("include"),
        "$torch_home/include",
        "$torch_home/include/torch/csrc/api/include",
        "$cuda_home/include",
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]
    system_includes += [p.resolve() for p in jit_env.CUTLASS_INCLUDE_DIRS]
    system_includes.append(jit_env.SPDLOG_INCLUDE_DIR.resolve())
    common_cflags = [
        "-DTORCH_EXTENSION_NAME=$name",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DPy_LIMITED_API=0x03090000",
    ]

    common_cflags += _get_pybind11_abi_build_flags()
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for dir in extra_include_dirs:
            common_cflags.append(f"-I{dir.resolve()}")

    is_windows = platform.system() == "Windows"

    if is_windows:
        for dir in system_includes:
            common_cflags.append(f"-I{dir}")
    else:
        for dir in system_includes:
            common_cflags.append(f"-isystem {dir}")

    cflags = [
        "$common_cflags",
    ]

    if not is_windows:
        cflags.append("-fPIC")

    if extra_cflags is not None:
        cflags += extra_cflags

    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", cc_env]


    common_cuda_flags = common_cflags.copy()

    if is_windows:
        common_cuda_flags = [
            "DTORCH_EXTENSION_NAME=$name",
            "-Xcompiler=/Zc:__cplusplus"
        ] + common_cuda_flags[1:]

    cuda_cflags += [
        "$common_cuda_flags",
        "--expt-relaxed-constexpr",
    ]
    if not is_windows:
        cuda_cflags.append("--compiler-options=-fPIC")

    cuda_cflags += _get_cuda_arch_flags(extra_cuda_cflags)
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags

    if is_windows:
        python_lib_path = os.path.join(sys.base_exec_prefix, "libs")
        ldflags = [
            f"/LIBPATH:{python_lib_path}",
            "/LIBPATH:$torch_home\\lib",
            "/LIBPATH:$cuda_home\\lib\\x64",
            "c10.lib",
            "c10_cuda.lib",
            "torch_cpu.lib",
            "torch_cuda.lib",
            "-INCLUDE:?warp_size@cuda@at@@YAHXZ",
            "torch.lib",
            "cudart.lib",
            "torch_python.lib"
        ]
    else:
        ldflags = [
            "-shared",
            "-L$torch_home/lib",
            "-lc10",
            "-lc10_cuda",
            "-ltorch_cpu",
            "-ltorch_cuda",
            "-ltorch",
            "-L$cuda_home/lib64",
            "-lcudart"
        ]

    env_extra_ldflags = os.environ.get("FLASHINFER_EXTRA_LDFLAGS")
    if env_extra_ldflags:
        try:
            import shlex

            ldflags += shlex.split(env_extra_ldflags)
        except ValueError as e:
            print(
                f"Warning: Could not parse FLASHINFER_EXTRA_LDFLAGS with shlex: {e}. Falling back to simple split.",
                file=sys.stderr,
            )
            ldflags += env_extra_ldflags.split()

    if extra_ldflags is not None:
        if is_windows:
            for ldflag in extra_ldflags:
                if ldflag.startswith("-l"):
                    ldflag = ldflag[2:] + ".lib"
                ldflags.append(ldflag)
        else:
            ldflags += extra_ldflags

    cxx = os.environ.get("CXX", "c++")
    cuda_home = CUDA_HOME or "/usr/local/cuda"
    nvcc = os.environ.get("PYTORCH_NVCC", "$cuda_home/bin/nvcc")

    if is_windows:
        rule_compile = [
            "rule compile",
            "  command = cl.exe $cflags -c $in /Fo$out $post_cflags",
            "  deps = msvc",
        ]
        rule_cuda_compile = [
            "rule cuda_compile",
            "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d -$cuda_cflags -c $in -o $out $cuda_post_cflags",
            "  depfile = $out.d",
            "  deps = msvc",
        ]
        rule_link = [
            "rule link",
            "  command = link.exe /DLL $in /nologo $ldflags /out:$out",
        ]
    else:
        rule_compile = [
            "rule compile",
            "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
        ]
        rule_cuda_compile = [
            "rule cuda_compile",
            "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
        ]
        rule_link = [
            "rule link",
            "  command = $cxx $in $ldflags -o $out",
        ]

    lines = [
        "ninja_required_version = 1.3",
        f"name = {name}",
        f"cuda_home = {cuda_home}",
        f"torch_home = {_TORCH_PATH}",
        f"cxx = {cxx}",
        f"nvcc = {nvcc}",
        "",
        "common_cflags = " + join_multiline(common_cflags),
        "common_cuda_flags = " + join_multiline(common_cuda_flags),
        "cflags = " + join_multiline(cflags),
        "post_cflags =",
        "cuda_cflags = " + join_multiline(cuda_cflags),
        "cuda_post_cflags =",
        "ldflags = " + join_multiline(ldflags),
        "",
        *rule_compile,
        "",
        *rule_cuda_compile,
        "",

    ]

    # Add nvcc linking rule for device code
    if needs_device_linking:
        lines.extend(
            [
                "rule nvcc_link",
                "  command = $nvcc -shared $in $ldflags -o $out",
                "",
            ]
        )
    else:
        lines.extend(
            [
                *rule_link,
                "",
            ]
        )

    objects = []
    for source in sources:
        is_cuda = source.suffix == ".cu"
        object_suffix = ".cuda.o" if is_cuda else ".o"
        cmd = "cuda_compile" if is_cuda else "compile"
        obj_name = source.with_suffix(object_suffix).name
        obj = f"$name/{obj_name}"
        objects.append(obj)
        source_path = source.resolve()
        if is_windows:
            source_path = str(source_path).replace(":\\", "$:\\")
        lines.append(f"build {obj}: {cmd} {source_path}")

    lines.append("")
    link_rule = "nvcc_link" if needs_device_linking else "link"
    if is_windows:
        lines.append(f"build $name.dll: {link_rule} " + " ".join(objects))
        lines.append("default $name.dll")
    else:
        lines.append(f"build $name/$name.so: {link_rule} " + " ".join(objects))
        lines.append("default $name/$name.so")
    lines.append("")

    return "\n".join(lines)


def run_ninja(workdir: Path, ninja_file: Path, verbose: bool) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command += ["-j", str(num_workers)]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e
