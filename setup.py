"""
Copyright (c) 2023 by FlashInfer team.

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

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping

import setuptools
from setuptools.dist import Distribution

root = Path(__file__).parent.resolve()
aot_ops_package_dir = root / "build" / "aot-ops-package-dir"

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    WIN_BUILD_DIR = Path("C:\\_fib") # short build dir to allow commands to stay below Windows max path length limit
    WIN_BUILD_DIR.mkdir(exist_ok=True)
    FAKE_BUILD_DIR = root / "build"
    if FAKE_BUILD_DIR.exists():
        if not FAKE_BUILD_DIR.is_symlink():
            shutil.rmtree(FAKE_BUILD_DIR, ignore_errors=True)
            FAKE_BUILD_DIR.symlink_to(WIN_BUILD_DIR)
    else:
        FAKE_BUILD_DIR.symlink_to(WIN_BUILD_DIR)

    if not aot_ops_package_dir.exists():
        aot_ops_package_dir.mkdir() # prevent pyproject.toml package-directory error

    if "--jit" in sys.argv: # from python setup.py bdist_wheel --jit, remove AOT libraries from the build
        sys.argv.remove("--jit")
        if aot_ops_package_dir.is_symlink():
            aot_ops_package_dir.unlink()
            aot_ops_package_dir.mkdir()
        lib_aot_dir = WIN_BUILD_DIR / "lib" / "flashinfer" / "data" / "aot"
        if lib_aot_dir.exists():
            shutil.rmtree(lib_aot_dir)

    enable_aot = aot_ops_package_dir.is_symlink()
else:
    enable_aot = aot_ops_package_dir.is_dir() and any(aot_ops_package_dir.iterdir())


def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def get_version():
    package_version = (root / "version.txt").read_text().strip()
    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION")
    if local_version is None:
        return package_version
    return f"{package_version}+{local_version}"


def generate_build_meta(aot_build_meta: dict) -> None:
    build_meta_str = f"__version__ = {get_version()!r}\n"
    if len(aot_build_meta) != 0:
        build_meta_str += f"build_meta = {aot_build_meta!r}\n"
    write_if_different(root / "flashinfer" / "_build_meta.py", build_meta_str)


ext_modules: List[setuptools.Extension] = []
cmdclass: Mapping[str, type[setuptools.Command]] = {}
install_requires = [
    "numpy",
    "torch",
    "ninja",
    "requests",
    "cuda-python<=12.9",
    "pynvml",
    "einops",
    "packaging>=24.2"
]
if not IS_WINDOWS:
    install_requires.append("nvidia-cudnn-frontend>=1.13.0")

generate_build_meta({})

if IS_WINDOWS:
    import distutils.command.build
    import distutils.command.clean

    # Custom build step
    class BuildCommand(distutils.command.build.build):
        def initialize_options(self):
            distutils.command.build.build.initialize_options(self)
            self.build_base = str(WIN_BUILD_DIR)
            self.build_temp = os.path.join(self.build_base, "t")

    # Custom clean step to remove short build dir
    class WindowsCleanCommand(distutils.command.clean.clean):
        def run(self):
            super().run()
            shutil.rmtree(WIN_BUILD_DIR, ignore_errors=True)

    cmdclass["build"] = BuildCommand
    cmdclass["clean"] = WindowsCleanCommand

if enable_aot:        
    import torch
    import torch.utils.cpp_extension as torch_cpp_ext
    from packaging.version import Version

    def get_cuda_version() -> Version:
        if torch_cpp_ext.CUDA_HOME is None:
            nvcc = "nvcc"
        else:
            nvcc = os.path.join(torch_cpp_ext.CUDA_HOME, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        return Version(re.findall(r"release (\d+\.\d+),", txt)[0])        

    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")

    if os.environ.get("FLASHINFER_USE_CXX11_ABI"):
        # force use cxx11 abi
        torch._C._GLIBCXX_USE_CXX11_ABI = 1

    cuda_version = get_cuda_version()
    torch_full_version = Version(torch.__version__)
    torch_version = f"{torch_full_version.major}.{torch_full_version.minor}"
    install_requires = [req for req in install_requires if not req.startswith("torch ")]
    install_requires.append(f"torch == {torch_version}.*")

    aot_build_meta = {}
    aot_build_meta["cuda_major"] = cuda_version.major
    aot_build_meta["cuda_minor"] = cuda_version.minor
    aot_build_meta["torch"] = torch_version
    aot_build_meta["python"] = platform.python_version()
    aot_build_meta["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
    generate_build_meta(aot_build_meta)


class AotDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return enable_aot


setuptools.setup(
    version=get_version(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    distclass=AotDistribution,
)