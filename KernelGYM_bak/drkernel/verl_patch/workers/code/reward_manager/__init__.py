# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

__all__ = [
    "CodeBinaryRewardManager",
    "CodeRewardManager",
    "HttpSandboxRewardManager",
    "NaiveRewardManager",
    "FileSearchRewardManager",
    "LocalSearchRewardManager",
    "SWERewardManager",
    "AsyncKernelRewardManager",
]


_MANAGER_MODULES = {
    "CodeBinaryRewardManager": (".code", "CodeBinaryRewardManager"),
    "CodeRewardManager": (".code", "CodeRewardManager"),
    "HttpSandboxRewardManager": (".http_sandbox", "HttpSandboxRewardManager"),
    "NaiveRewardManager": (".naive", "NaiveRewardManager"),
    "FileSearchRewardManager": (".search.file_search", "FileSearchRewardManager"),
    "LocalSearchRewardManager": (".search.local_search", "LocalSearchRewardManager"),
    "SWERewardManager": (".swe", "SWERewardManager"),
    "AsyncKernelRewardManager": ("kernel.workers.reward_manager.kernel_async", "AsyncKernelRewardManager"),
}


def __getattr__(name):
    if name not in _MANAGER_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _MANAGER_MODULES[name]
    module = importlib.import_module(module_name, __name__ if module_name.startswith(".") else None)
    obj = getattr(module, attr_name)
    globals()[name] = obj
    return obj
