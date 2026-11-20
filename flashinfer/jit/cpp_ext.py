# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py

import functools
import hashlib
import logging
import os
import re
import subprocess
import sys
import sysconfig
from packaging.version import Version
from pathlib import Path
from typing import List, Mapping, Optional

import tvm_ffi
import torch

from . import env as jit_env
from ..compilation_context import CompilationContext

is_windows = os.name == "nt"
logger = logging.getLogger(__name__)


def parse_env_flags(env_var_name) -> List[str]:
    env_flags = os.environ.get(env_var_name)
    if env_flags:
        try:
            import shlex

            return shlex.split(env_flags)
        except ValueError as e:
            logger.warning(
                "Could not parse %s with shlex: %s. Falling back to simple split.",
                env_var_name,
                e,
            )
            return env_flags.split()
    return []


def _get_glibcxx_abi_build_flags() -> List[str]:
    glibcxx_abi_cflags = [] if is_windows else [
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    ]
    return glibcxx_abi_cflags


@functools.cache
def get_cuda_path() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return cuda_home
    # get output of "which nvcc"
    nvcc_path = subprocess.run(["which", "nvcc"], capture_output=True)
    if nvcc_path.returncode == 0:
        cuda_home = os.path.dirname(
            os.path.dirname(nvcc_path.stdout.decode("utf-8").strip())
        )
    else:
        cuda_home = "/usr/local/cuda"  # This default value is from: https://github.com/pytorch/pytorch/blob/ceb11a584d6b3fdc600358577d9bf2644f88def9/torch/utils/cpp_extension.py#L115
        if not os.path.exists(cuda_home):
            raise RuntimeError(
                f"Could not find nvcc and default {cuda_home=} doesn't exist"
            )
    return cuda_home


@functools.cache
def get_cuda_version() -> Version:
    # Try to query nvcc for CUDA version; if nvcc is unavailable, fall back to torch.version.cuda
    try:
        cuda_home = get_cuda_path()
        nvcc = os.path.join(cuda_home, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        matches = re.findall(r"release (\d+\.\d+),", txt)
        if not matches:
            raise RuntimeError(
                f"Could not parse CUDA version from nvcc --version output: {txt}"
            )
        return Version(matches[0])
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        # NOTE(Zihao): when nvcc is unavailable, fall back to torch.version.cuda
        if torch.version.cuda is None:
            raise RuntimeError(
                "nvcc not found and PyTorch is not built with CUDA support. "
                "Could not determine CUDA version."
            ) from e
        return Version(torch.version.cuda)


def is_cuda_version_at_least(version_str: str) -> bool:
    return get_cuda_version() >= Version(version_str)


def get_nvcc_parallelism_flags() -> List[str]:
    """Build nvcc flags controlled by FlashInfer parallelism environment variables."""
    env_var_name = "FLASHINFER_NVCC_THREADS"
    default = 1
    value = os.environ.get(env_var_name, str(default))

    try:
        threads = int(value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %s.", env_var_name, value, default
        )
        threads = default

    if threads < 1:
        logger.warning("Ignoring %s=%r; value must be >= 1.", env_var_name, value)
        threads = default

    return [f"--threads={threads}"]


@functools.lru_cache(maxsize=1)
def host_compiler_is_gcc() -> bool:
    """Whether the compiler nvcc forwards -Xcompiler flags to is GCC.

    Cached because the underlying torch helper shells out to the compiler.
    """
    from torch.utils.cpp_extension import check_compiler_is_gcc, get_cxx_compiler

    host_compiler = os.environ.get("CC", get_cxx_compiler())
    return check_compiler_is_gcc(host_compiler)


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def get_cccl_includes() -> List:
    """Get vendored CCCL include directories (added with -I for CTK override precedence)."""
    return [p.resolve() for p in jit_env.CCCL_INCLUDE_DIRS]


def get_system_includes(cuda_home: str) -> List:
    """Get list of system include directories."""
    system_includes = [
        sysconfig.get_path("include"),
        "$cuda_home/include",
        tvm_ffi.libinfo.find_include_path(),
        tvm_ffi.libinfo.find_dlpack_include_path(),
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]
    system_includes += [p.resolve() for p in jit_env.CUTLASS_INCLUDE_DIRS]
    system_includes.append(jit_env.SPDLOG_INCLUDE_DIR.resolve())

    if cuda_home == "/usr":
        # NOTE: this will resolve to /usr/include, which will mess up includes. See #1793
        system_includes.remove("$cuda_home/include")

    return system_includes


def build_common_cflags(
    cuda_home: str,
    extra_include_dirs: Optional[List[Path]] = None,
) -> List[str]:
    """Build common compilation flags."""
    cccl_includes = get_cccl_includes()
    system_includes = get_system_includes(cuda_home)

    common_cflags = []
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        common_cflags.append("-DPy_LIMITED_API=0x03090000")
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for extra_dir in extra_include_dirs:
            if is_windows:
                # Quote: torch's include_paths() returns CUDA/cuDNN dirs under
                # "C:\Program Files\...", and an unquoted space splits the flag
                # into stray input files ("nvcc fatal: A single input file is
                # required for a non-link phase ...").
                common_cflags.append(f'-I"{extra_dir.resolve()}"')
            else:
                common_cflags.append(f"-I{extra_dir.resolve()}")
    # Vendored CCCL headers use -I (not -isystem) so they take precedence
    # over the CTK-bundled copy. CCCL headers use #pragma system_header
    # internally to suppress warnings. See https://github.com/NVIDIA/cccl/issues/527
    if is_windows:
        for cccl_dir in cccl_includes:
            common_cflags.append(f'-I"{str(cccl_dir)}"')
        for sys_dir in system_includes:
            common_cflags.append(f'-I"{str(sys_dir)}"')
    else:
        for cccl_dir in cccl_includes:
            common_cflags.append(f"-I{cccl_dir}")
        for sys_dir in system_includes:
            common_cflags.append(f"-isystem {sys_dir}")

    return common_cflags


def build_cflags(
    common_cflags: List[str],
    extra_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build C++ compilation flags."""
    cflags = [
        "$common_cflags",
    ]

    if not is_windows:
        cflags.append("-fPIC")
    else:
        cflags.append("/std:c++20")
        cflags.append("/DNOMINMAX")
        cflags.append("/Zc:preprocessor")

    if extra_cflags is not None:
        cflags += extra_cflags

    env_extra_cflags = parse_env_flags("FLASHINFER_EXTRA_CFLAGS")
    if env_extra_cflags is not None:
        cflags += env_extra_cflags

    cflags = list(set(cflags))

    return cflags


def build_cuda_cflags(
    common_cflags: List[str],
    extra_cuda_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build CUDA compilation flags."""
    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", f'"{cc_env}"']
    common_cuda_flags  = common_cflags.copy()

    if is_windows:
        common_cuda_flags = [
            "-DTORCH_EXTENSION_NAME=$name",
            "--std=c++20",
            "-Xcompiler /Zc:__cplusplus",
            "-Xcompiler /Zc:preprocessor",
        ] + [f for f in common_cuda_flags if not f.startswith("-DPy_LIMITED_API")]

    cuda_cflags += [
        "$common_cuda_flags",
        "--expt-relaxed-constexpr",
    ]

    if not is_windows:
        cuda_cflags.append("--compiler-options=-fPIC")
    cuda_version = get_cuda_version()
    # enable -static-global-template-stub when cuda version >= 12.8
    if cuda_version >= Version("12.8"):
        cuda_cflags += [
            "-static-global-template-stub=false",
        ]

    cpp_ext_initial_compilation_context = CompilationContext()
    global_flags = cpp_ext_initial_compilation_context.get_nvcc_flags_list(
        map_sm107_to_100f=True
    )
    if extra_cuda_cflags is not None:
        # Check if module provides architecture flags
        module_has_gencode = any(
            flag.startswith("-gencode=") for flag in extra_cuda_cflags
        )

        if module_has_gencode:
            # Use module's architecture flags, but keep global non-architecture flags
            global_non_arch_flags = [
                flag for flag in global_flags if not flag.startswith("-gencode=")
            ]
            cuda_cflags += global_non_arch_flags + extra_cuda_cflags
        else:
            # No module architecture flags, use both global and module flags
            cuda_cflags += global_flags + extra_cuda_cflags
    else:
        # No module flags, use global flags
        cuda_cflags += global_flags

    env_extra_cuda_cflags = parse_env_flags("FLASHINFER_EXTRA_CUDAFLAGS")
    if env_extra_cuda_cflags is not None:
        cuda_cflags += env_extra_cuda_cflags

    return cuda_cflags, common_cuda_flags


@functools.cache
def get_tvm_ffi_lib_dir() -> Path:
    """Folder holding tvm_ffi's import library (tvm_ffi.lib on Windows).

    Asks tvm_ffi where its library is, instead of assuming
    <python>\\Lib\\site-packages, which is wrong for `pip install --user`,
    Microsoft Store Python and other layouts.
    """
    find_implib = getattr(tvm_ffi.libinfo, "find_windows_implib", None)
    if is_windows and find_implib is not None:
        try:
            return Path(find_implib()).parent
        except RuntimeError:
            pass  # older layouts: fall back to the DLL's folder
    return Path(tvm_ffi.libinfo.find_libtvm_ffi()).parent


@functools.cache
def get_torch_lib_dir() -> Path:
    """Folder holding torch's libraries (c10.lib, torch.lib, ... on Windows)."""
    return Path(torch.__file__).resolve().parent / "lib"


def ninja_escape_path(path) -> str:
    """Escape a path for use in a ninja ``build`` line.

    On Windows the drive colon must be written ``$:``. Spaces must be ``$ ``
    on every platform.
    """
    s = str(path)
    if is_windows:
        s = s.replace(":\\", "$:\\").replace(" ", "$ ")
    return s


def get_object_name(source: Path, sources: List[Path]) -> str:
    """Object file name for ``source`` inside the spec's build directory.

    Linux keeps the upstream ``<parent>_<stem>`` scheme. On Windows the parent
    of generated sources is the spec directory itself (named after the spec),
    and the object already lives in ``<jit_dir>/<spec name>/``, so the prefix
    repeats the spec name in the path and pushes objects and their ``.d``
    depfiles past MAX_PATH (259 chars). There only the stem is used. If another
    source of the same kind in this spec has the same stem, a short hash of the
    parent directory is added to keep the names unique.
    """
    is_cuda = source.suffix == ".cu"
    object_suffix = ".cuda.o" if is_cuda else ".o"
    if not is_windows:
        return f"{source.parent.name}_{source.stem}{object_suffix}"

    parent = Path(source).resolve().parent
    clashes = any(
        Path(other).stem == source.stem
        and (Path(other).suffix == ".cu") == is_cuda
        and Path(other).resolve().parent != parent
        for other in sources
    )
    if clashes:
        digest = hashlib.sha256(str(parent).lower().encode()).hexdigest()[:8]
        return f"{source.stem}_{digest}{object_suffix}"
    return f"{source.stem}{object_suffix}"


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
    needs_device_linking: bool = False,
    embedded_cubins: Optional[Mapping[str, Path]] = None,
    extra_cuda_cflags_by_source: Optional[Mapping[Path, List[str]]] = None,
) -> str:
    cuda_home = get_cuda_path()
    common_cflags = build_common_cflags(cuda_home, extra_include_dirs)
    cflags = build_cflags(common_cflags, extra_cflags)
    cuda_cflags, common_cuda_flags = build_cuda_cflags(common_cflags, extra_cuda_cflags)
    cuda_arch_flags = [flag for flag in cuda_cflags if flag.startswith("-gencode=")]
    # build_cuda_cflags returns (cuda_cflags, common_cuda_flags); only the first
    # is per-source. common_cuda_flags depends on common_cflags alone, so the
    # module-level $common_cuda_flags variable stays valid for every source.
    cuda_cflags_by_source = {
        Path(source).resolve(): build_cuda_cflags(common_cflags, flags)[0]
        for source, flags in (extra_cuda_cflags_by_source or {}).items()
    }

    if is_windows:
        python_lib_path = os.path.join(sys.base_exec_prefix, "libs")
        plat = sysconfig.get_platform().upper()  # e.g. "win-arm64", "win-amd64"
        win_arch = "x64"
        if plat.endswith("ARM64") or plat.endswith("AARCH64"):
            win_arch = "arm64"
        ldflags = [
            f'"/LIBPATH:{python_lib_path}"',
            f'"/LIBPATH:$cuda_home\\lib\\{win_arch}"',
            f'"/LIBPATH:{get_tvm_ffi_lib_dir()}"',
            f'"/LIBPATH:{get_torch_lib_dir()}"',
            "c10.lib",
            "c10_cuda.lib",
            "torch.lib",
            "torch_cuda.lib",
            "cudart.lib",
            "cuda.lib",
            "tvm_ffi.lib",
            "torch_python.lib"
        ]
    else:
        ldflags = [
            "-shared",
            "-L$cuda_home/lib64",
            "-L$cuda_home/lib64/stubs",
            "-lcudart",
            "-lcuda",
        ]

    env_extra_ldflags = parse_env_flags("FLASHINFER_EXTRA_LDFLAGS")
    if env_extra_ldflags is not None:
        ldflags += env_extra_ldflags

    if extra_ldflags is not None:
        if is_windows:
            for ldflag in extra_ldflags:
                if ldflag.startswith("-l"):
                    ldflag = ldflag[2:] + ".lib"
                ldflags.append(ldflag)
        else:
            ldflags += extra_ldflags

    cxx = os.environ.get("CXX", "c++")
    nvcc = os.environ.get("FLASHINFER_NVCC", "$cuda_home/bin/nvcc")

    if is_windows:
        nvcc = f'"{nvcc}"'
    # Compiler launchers (e.g., sccache, ccache) — empty string when unset
    cxx_launcher = os.environ.get("FLASHINFER_CXX_LAUNCHER", "")
    nvcc_launcher = os.environ.get("FLASHINFER_NVCC_LAUNCHER", "")

    if is_windows:
        rule_compile = [
            "rule compile",
            "  command = cl.exe $cflags -c $in /Fo$out $post_cflags",
            "  deps = msvc",
        ]
        rule_cuda_compile = [
            "rule cuda_compile",
            "  command = $nvcc --generate-dependencies-with-compile -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
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
            "  command = $cxx_launcher $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
        ]
        rule_cuda_compile = [
            "rule cuda_compile",
            "  command = $nvcc_launcher $nvcc --generate-dependencies-with-compile -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
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
        f"cxx = {cxx}",
        f"nvcc = {nvcc}",
        f"cxx_launcher = {cxx_launcher}",
        f"nvcc_launcher = {nvcc_launcher}",
        "",
        "common_cflags = " + join_multiline(common_cflags),
        "common_cuda_flags = " + join_multiline(common_cuda_flags),
        "cflags = " + join_multiline(cflags),
        "post_cflags =",
        "cuda_cflags = " + join_multiline(cuda_cflags),
        "cuda_post_cflags =",
        "cuda_arch_flags = " + join_multiline(cuda_arch_flags),
        "ldflags = " + join_multiline(ldflags),
        "",
        *rule_compile,
        "",
        *rule_cuda_compile,
        "",
    ]

    if embedded_cubins:
        lines.extend(
            [
                "rule embed_cubin",
                f'  command = "{sys.executable}" -m tvm_ffi.utils.embed_cubin '
                '--output-obj "$out" --input-obj "$in" --cubin "$cubin" '
                "--name $cubin_name",
                "",
            ]
        )

    # Add nvcc linking rule for device code
    if needs_device_linking:
        lines.extend(
            [
                "rule nvcc_link",
                "  command = $nvcc -shared $cuda_arch_flags $in $ldflags -o $out",
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

    # Use absolute paths for outputs so ninja files work with any workdir
    # This enables isolated workdirs for runtime JIT (avoiding .ninja_log races)
    # while still supporting subninja for parallel AOT builds
    output_dir = jit_env.FLASHINFER_JIT_DIR / name

    objects = []
    for source in sources:
        is_cuda = source.suffix == ".cu"
        object_suffix = ".cuda.o" if is_cuda else ".o"
        cmd = "cuda_compile" if is_cuda else "compile"
        obj_name = get_object_name(source, sources)
        obj = ninja_escape_path((output_dir / obj_name).resolve())
        objects.append(obj)
        resolved_source = source.resolve()
        source_path = ninja_escape_path(resolved_source)
        lines.append(f"build {obj}: {cmd} {source_path}")
        if resolved_source in cuda_cflags_by_source:
            lines.append(
                "  cuda_cflags = "
                + join_multiline(cuda_cflags_by_source[resolved_source])
            )

    if embedded_cubins:
        if not objects:
            raise ValueError("embedded cubins require at least one host object")
        # objects[] already holds ninja-escaped paths; keep the base name from
        # the unescaped source so the chain names stay short (embedded_<i>_<obj>
        # rather than growing embedded_1_embedded_0_... with each cubin).
        current_obj = objects[0]
        base_obj_name = get_object_name(sources[0], sources)
        for index, (cubin_name, cubin_path) in enumerate(
            sorted(embedded_cubins.items())
        ):
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cubin_name) is None:
                raise ValueError(f"invalid embedded cubin identifier: {cubin_name!r}")
            resolved_cubin = Path(cubin_path).resolve(strict=True)
            embedded_obj = ninja_escape_path(
                (output_dir / f"embedded_{index}_{base_obj_name}").resolve()
            )
            lines.extend(
                [
                    f"build {embedded_obj}: embed_cubin {current_obj} | "
                    f"{ninja_escape_path(resolved_cubin)}",
                    # Variable values need no colon/space escaping; "$" is the
                    # only special character there and paths do not contain it.
                    f"  cubin = {resolved_cubin}",
                    f"  cubin_name = {cubin_name}",
                ]
            )
            current_obj = embedded_obj
        objects[0] = current_obj

    lines.append("")
    link_rule = "nvcc_link" if needs_device_linking else "link"
    if is_windows:
        output_so = ninja_escape_path((output_dir / f"{name}.dll").resolve())
        lines.append(f"build {output_so}: {link_rule} " + " ".join(objects))
        lines.append(f"default {output_so}")
    else:
        output_so = ninja_escape_path((output_dir / f"{name}.so").resolve())
        lines.append(f"build {output_so}: {link_rule} " + " ".join(objects))
        lines.append(f"default {output_so}")
    lines.append("")

    return "\n".join(lines)


def _get_num_workers() -> Optional[int]:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit():
        return int(max_jobs)
    return None


def run_ninja(
    workdir: Path,
    ninja_file: Path,
    verbose: bool,
    max_jobs: Optional[int] = None,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = max_jobs if max_jobs is not None else _get_num_workers()
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
