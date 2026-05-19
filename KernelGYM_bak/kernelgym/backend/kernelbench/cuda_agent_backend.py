"""CUDA-Agent backend implementation for KernelBench.

This backend compiles and executes raw CUDA code using the CUDA-Agent workflow:
1. Write CUDA source files (.cu, .cpp) to a working directory
2. Compile using torch.utils.cpp_extension
3. Load the compiled shared library and execute
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.utils.cpp_extension as cpp_ext

from kernelgym.backend.base import Backend
from kernelgym.toolkit.kernelbench.loading import graceful_eval_cleanup
from kernelgym.toolkit.validation import validate_code


class CudaAgentBackendSession:
    """Session for a loaded CUDA-Agent backend."""

    def __init__(
        self,
        handle: Dict[str, Any],
        device: torch.device,
    ):
        self.handle = handle
        self.device = device
        self.model_cls = handle.get("model_cls")
        self.work_dir = handle.get("work_dir")
        self.so_path = handle.get("so_path")

    def create_model(self, init_inputs: Any, no_grad: bool = True, synchronize: bool = False):
        """Create a model instance."""
        if self.model_cls is None:
            raise ValueError("Model class not loaded")

        # Move inputs to device
        if isinstance(init_inputs, list):
            init_inputs = [
                x.cuda(device=self.device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]
        elif isinstance(init_inputs, dict):
            init_inputs = {
                k: v.cuda(device=self.device) if isinstance(v, torch.Tensor) else v
                for k, v in init_inputs.items()
            }

        if no_grad:
            with torch.no_grad():
                if isinstance(init_inputs, dict):
                    model = self.model_cls(**init_inputs)
                else:
                    model = self.model_cls(*init_inputs)
        else:
            if isinstance(init_inputs, dict):
                model = self.model_cls(**init_inputs)
            else:
                model = self.model_cls(*init_inputs)

        if hasattr(model, "to"):
            model = model.to(self.device)

        if synchronize and self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)

        return model

    def close(self):
        """Close the session and clean up."""
        if self.work_dir and Path(self.work_dir).exists():
            try:
                shutil.rmtree(self.work_dir)
            except Exception:
                pass


class CudaAgentBackend(Backend):
    """Backend for compiling and running raw CUDA code using CUDA-Agent workflow."""

    name = "kernelbench.cuda_agent"

    def __init__(self):
        self._work_dirs: list[Path] = []

    def _normalize_device(self, device: Any | None) -> torch.device:
        if device is None:
            return torch.device("cuda:0")
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    def _maybe_set_cuda_device(self, device: torch.device) -> None:
        if device.type != "cuda":
            return
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass

    def _create_work_dir(self) -> Path:
        """Create a temporary working directory for CUDA compilation."""
        work_dir = Path(tempfile.mkdtemp(prefix="cuda_agent_"))
        self._work_dirs.append(work_dir)
        return work_dir

    def _setup_cuda_project(self, work_dir: Path, code: str, entry_point: str = "ModelNew"):
        """Set up the CUDA project structure in the working directory.

        Args:
            work_dir: The working directory to set up
            code: The Python model code that uses the CUDA extension
            entry_point: The name of the model class (default: ModelNew)
        """
        # Create kernels directory
        kernels_dir = work_dir / "kernels"
        kernels_dir.mkdir(exist_ok=True)

        # Write the Python model file
        model_file = work_dir / "model_new.py"
        model_file.write_text(code)

        # Create a default binding.cpp
        binding_cpp = work_dir / "binding.cpp"
        binding_cpp_content = '''#include <pybind11/pybind11.h>
#include "binding_registry.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BindingRegistry::getInstance().applyBindings(m);
}
'''
        binding_cpp.write_text(binding_cpp_content)

        # Create binding_registry.h
        binding_registry_h = work_dir / "binding_registry.h"
        binding_registry_content = '''#pragma once

#include <vector>
#include <functional>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class BindingRegistry {
public:
    using BindingFunction = std::function<void(pybind11::module&)>;

    static BindingRegistry& getInstance() {
        static BindingRegistry instance;
        return instance;
    }

    void registerBinding(const std::string& name, BindingFunction func) {
        bindings_.push_back({name, func});
    }

    void applyBindings(pybind11::module& m) {
        for (auto& [name, func] : bindings_) {
            func(m);
        }
    }

private:
    std::vector<std::pair<std::string, BindingFunction>> bindings_;
    BindingRegistry() = default;
};

class BindingRegistrar {
public:
    BindingRegistrar(const std::string& name, BindingRegistry::BindingFunction func) {
        BindingRegistry::getInstance().registerBinding(name, func);
    }
};

#define REGISTER_BINDING(name, func) \\
    static BindingRegistrar _registrar_##name(#name, [](pybind11::module& m) { func(m); })
'''
        binding_registry_h.write_text(binding_registry_content)

    def _extract_cuda_sources(self, work_dir: Path) -> list[str]:
        """Extract CUDA source files from the working directory.

        Args:
            work_dir: The working directory

        Returns:
            List of source file paths
        """
        kernels_dir = work_dir / "kernels"
        sources = []

        # Find all .cu and .cpp files in root and kernels directory
        root_sources = list(work_dir.glob("*.cu")) + list(work_dir.glob("*.cpp"))
        kernel_sources = []
        if kernels_dir.is_dir():
            kernel_sources = list(kernels_dir.glob("*.cu")) + list(kernels_dir.glob("*.cpp"))

        sources = sorted(set([str(s) for s in root_sources + kernel_sources]))

        return sources

    def _compile_cuda(self, work_dir: Path, sources: list[str]) -> Dict[str, Any]:
        """Compile CUDA sources using torch.utils.cpp_extension.

        Args:
            work_dir: The working directory
            sources: List of source file paths

        Returns:
            Dictionary with compilation results
        """
        if not sources:
            return {
                "compiled": False,
                "error": "No CUDA source files found (*.cu, *.cpp)",
            }

        build_dir = work_dir / "build" / "forced_compile"
        output_so = work_dir / "cuda_extension.so"

        # # 2. 【关键修复】清理全局 PyTorch 扩展缓存，防止 _v1, _v2 后缀产生
        # # torch 默认缓存位置: ~/.cache/torch_extensions/cuda_extension
        # global_cache_dir = Path.home() / ".cache" / "torch_extensions" / "cuda_extension"
        # if global_cache_dir.exists():
        #     try:
        #         shutil.rmtree(global_cache_dir)
        #         print(f"[CudaAgent] Cleaned global cache: {global_cache_dir}")
        #     except Exception as e:
        #         print(f"[CudaAgent] Warning: Could not clean global cache: {e}")

        # Clean up previous builds
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        if output_so.exists():
            output_so.unlink()

        try:
            # Use torch.utils.cpp_extension to compile
            # Generate a unique extension name based on the work_dir to avoid torch cache conflicts
            # and file lock contentions during multiprocessing.
            ext_name = work_dir.name.replace("-", "_")

            print(f"Compiling CUDA sources in {str(build_dir)} with name {ext_name}")
            
            module = cpp_ext.load(
                name=ext_name,
                sources=sources,
                build_directory=str(build_dir),
                verbose=False,
                with_cuda=True,
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
            )

            # Get actual module name and file path
            module_name = module.__name__.split('.')[-1]
            if hasattr(module, '__file__'):
                built_so = Path(module.__file__)
            else:
                # Fallback to default behavior if __file__ is missing
                built_so = build_dir / "cuda_extension.so"
            
            print(f"Built so path: {built_so}")
            print(f"Output so path: {output_so}")
            # Copy the compiled .so file to the work directory
            # We use the original name "cuda_extension.so" for the output file
            # to maintain consistency, but we return the actual module name
            if built_so.exists():
                shutil.copy2(built_so, output_so)
                return {
                    "compiled": True,
                    "so_path": str(output_so),
                    "module_name": module_name,
                    "error": None,
                }
            else:
                return {
                    "compiled": False,
                    "error": "Compilation finished but .so file was not generated",
                }

        except Exception as exc:
            return {
                "compiled": False,
                "error": str(exc),
            }

    def _parse_cuda_sources_from_code(self, code: str) -> tuple[dict[str, str], str]:
        """Parse CUDA sources embedded in the code string.
        
        The code string can contain CUDA sources in the format:
        ### CUDA_SOURCES ###
        {'filename.cu': 'content', ...}
        ### END_CUDA_SOURCES ###
        
        Args:
            code: The code string that may contain embedded CUDA sources
            
        Returns:
            Tuple of (cuda_sources dict, python_code without cuda sources)
        """
        import ast
        
        cuda_sources = {}
        python_code = code
        
        # Look for CUDA sources section
        start_marker = "### CUDA_SOURCES ###"
        end_marker = "### END_CUDA_SOURCES ###"
        
        start_idx = code.find(start_marker)
        end_idx = code.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            # Extract the CUDA sources section
            sources_section = code[start_idx + len(start_marker):end_idx].strip()
            
            # Try to parse as a Python dict
            try:
                cuda_sources = ast.literal_eval(sources_section)
                if not isinstance(cuda_sources, dict):
                    cuda_sources = {}
            except (ValueError, SyntaxError):
                # If parsing fails, treat as empty
                cuda_sources = {}
            
            # Remove the CUDA sources section from the code
            python_code = code[:start_idx] + code[end_idx + len(end_marker):]
        
        return cuda_sources, python_code.strip()

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        """Compile CUDA code.

        Args:
            code: The Python model code that imports and uses cuda_extension.
                  Can also contain embedded CUDA sources in the format:
                  ### CUDA_SOURCES ###
                  {'filename.cu': 'content', ...}
                  ### END_CUDA_SOURCES ###
            **kwargs: Additional arguments including:
                - device: The CUDA device to use
                - entry_point: The name of the model class (default: ModelNew)
                - work_dir: Optional working directory (created if not provided)
                - cuda_sources: Dict of {filename: content} for CUDA source files

        Returns:
            Dictionary with compilation results
        """
        device = self._normalize_device(kwargs.get("device"))
        entry_point = kwargs.get("entry_point", "ModelNew")
        work_dir = kwargs.get("work_dir")
        cuda_sources = kwargs.get("cuda_sources", {})  # Dict of {filename: content}

        # Parse CUDA sources from code if embedded
        embedded_sources, python_code = self._parse_cuda_sources_from_code(code)
        if embedded_sources:
            cuda_sources.update(embedded_sources)
            code = python_code

        # Validate the Python code
        valid, error = validate_code(code, entry_point)
        if not valid:
            return {
                "compiled": False,
                "error": error,
                "device": str(device),
                "entry_point": entry_point,
                "backend": "cuda_agent",
            }

        # Create working directory
        if work_dir is None:
            work_dir = self._create_work_dir()
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        # Set up the CUDA project structure
        self._setup_cuda_project(work_dir, code, entry_point)

        # Write CUDA source files if provided
        kernels_dir = work_dir / "kernels"
        for filename, content in cuda_sources.items():
            file_path = kernels_dir / filename
            file_path.write_text(content)

        # Extract CUDA sources
        sources = self._extract_cuda_sources(work_dir)

        if not sources:
            return {
                "compiled": False,
                "error": "No CUDA source files found. Please provide cuda_sources with .cu files.",
                "device": str(device),
                "entry_point": entry_point,
                "backend": "cuda_agent",
                "work_dir": str(work_dir),
            }

        # Compile CUDA sources
        result = self._compile_cuda(work_dir, sources)

        artifact = {
            "compiled": result["compiled"],
            "error": result.get("error"),
            "device": str(device),
            "entry_point": entry_point,
            "backend": "cuda_agent",
            "work_dir": str(work_dir),
            "so_path": result.get("so_path"),
            "module_name": result.get("module_name", "cuda_extension"),
            "code": code,
        }

        return artifact

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        """Load the compiled CUDA extension and model.

        Args:
            artifact: The compilation artifact from compile()
            **kwargs: Additional arguments including:
                - device: The CUDA device to use
                - context: The execution context

        Returns:
            Handle containing the loaded model class and related info
        """
        code = artifact.get("code")
        entry_point = artifact.get("entry_point", "ModelNew")
        work_dir = artifact.get("work_dir")
        so_path = artifact.get("so_path")
        module_name = artifact.get("module_name", "cuda_extension")
        context = kwargs.get("context") or {}

        if not code:
            raise ValueError("CudaAgentBackend.load requires kernel code in artifact")

        if not so_path or not Path(so_path).exists():
            raise ValueError(f"Compiled shared library not found: {so_path}")

        device = self._normalize_device(kwargs.get("device") or artifact.get("device"))
        self._maybe_set_cuda_device(device)

        work_dir_path = Path(work_dir)
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

        # Import the model module
        import importlib.util

        model_file = work_dir_path / "model_new.py"
        spec = importlib.util.spec_from_file_location("cuda_agent_model", str(model_file))
        model_module = importlib.util.module_from_spec(spec)

        # Clean up old cuda_extension from sys.modules to avoid conflicts
        if "cuda_extension" in sys.modules:
            del sys.modules["cuda_extension"]

        import importlib.machinery
        # We must load using the actual module_name (e.g. cuda_extension_v1) so PyInit_xxx matches
        loader = importlib.machinery.ExtensionFileLoader(module_name, so_path)
        cuda_ext_module = loader.load_module()
        # Inject into sys.modules under the name the model expects
        sys.modules["cuda_extension"] = cuda_ext_module

        # Now load the model
        spec.loader.exec_module(model_module)

        model_cls = getattr(model_module, entry_point)

        if model_cls is None:
            raise ValueError(f"Failed to load model class '{entry_point}' from code")

        return {
            "model_cls": model_cls,
            "work_dir": work_dir,
            "so_path": so_path,
            "context": context,
            "backend": "cuda_agent",
            "entry_point": entry_point,
            "device": device,
            "tempfile_handle": None,
        }

    def create_model(self, handle: Any, init_inputs: Any, **kwargs: Any) -> Any:
        """Create an instance of the model.
        
        Args:
            handle: Handle returned by load()
            init_inputs: List of inputs for model initialization
            kwargs: Additional arguments
            
        Returns:
            Instantiated PyTorch model
        """
        if not isinstance(handle, dict) or "model_cls" not in handle:
            raise ValueError("CudaAgentBackend.create_model expects a handle from load()")
            
        device = self._normalize_device(kwargs.get("device") or handle.get("device"))
        session = CudaAgentBackendSession(handle, device)
        
        no_grad = kwargs.get("no_grad", True)
        synchronize = kwargs.get("synchronize", False)
        
        return session.create_model(init_inputs, no_grad=no_grad, synchronize=synchronize)

    def open_session(self, handle: Dict[str, Any], **kwargs: Any) -> CudaAgentBackendSession:
        """Open a session for the loaded backend.

        Args:
            handle: The handle from load()
            **kwargs: Additional arguments including:
                - device: The CUDA device to use

        Returns:
            CudaAgentBackendSession instance
        """
        device = self._normalize_device(kwargs.get("device") or handle.get("device"))
        return CudaAgentBackendSession(handle, device)

    def run(self, handle: Any, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Execute the model and return runtime metrics.

        Args:
            handle: The handle from load()
            inputs: Dictionary containing inputs for the model
            **kwargs: Additional arguments

        Returns:
            Dictionary with execution results
        """
        if not isinstance(handle, dict) or "model_cls" not in handle:
            raise ValueError("CudaAgentBackend.run expects a handle from load()")

        device = self._normalize_device(kwargs.get("device") or handle.get("device"))
        self._maybe_set_cuda_device(device)

        init_inputs = inputs.get("init_inputs", inputs.get("inputs", []))
        run_inputs = inputs.get("inputs", init_inputs)

        # Move inputs to device
        if isinstance(run_inputs, list):
            run_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in run_inputs
            ]
        elif isinstance(run_inputs, dict):
            run_inputs = {
                k: v.cuda(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in run_inputs.items()
            }

        # Create model
        model_cls = handle["model_cls"]
        if isinstance(init_inputs, dict):
            model = model_cls(**init_inputs)
        else:
            model = model_cls(*init_inputs)

        if hasattr(model, "to"):
            model = model.to(device)

        # Run inference
        with torch.no_grad():
            output = (
                model(**run_inputs)
                if isinstance(run_inputs, dict)
                else model(*run_inputs)
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

        return {"output": output}

    def cleanup(self, handle: Any, **kwargs: Any) -> None:
        """Clean up resources.

        Args:
            handle: The handle from load()
            **kwargs: Additional arguments
        """
        if not isinstance(handle, dict):
            return

        work_dir = handle.get("work_dir")
        if work_dir and Path(work_dir).exists():
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass

    def close(self, handle: Any, **kwargs: Any) -> None:
        """Close the backend and clean up."""
        self.cleanup(handle, **kwargs)
