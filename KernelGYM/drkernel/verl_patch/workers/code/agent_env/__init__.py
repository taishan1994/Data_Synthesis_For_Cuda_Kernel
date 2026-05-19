"""
Agent environment module with base classes and implementations.

This module ensures that all exec_tool_call methods in environments
inheriting from BaseEnv are decorated with with_timeout_and_retry.
"""

import importlib
import inspect

from .base_env import BaseEnv, FinishReasonTypeEnum, with_timeout_and_retry

__all__ = [
    "BaseEnv",
    "MathSandboxEnv",
    "CodeSandboxEnv",
    "LocalSearchEnv",
    "FileSearchEnv",
    "SWEFileLocationEnv",
    "with_timeout_and_retry",
    "FinishReasonTypeEnum",
    "create_environment",
]


_ENV_MODULES = {
    "MathSandboxEnv": ".math_sandbox_env",
    "CodeSandboxEnv": ".code_sandbox_env",
    "LocalSearchEnv": ".local_search_env",
    "FileSearchEnv": ".file_search_env",
    "SWEFileLocationEnv": ".swe_file_location_env",
}


def _load_env_class(env_type: str):
    if env_type not in _ENV_MODULES:
        raise ValueError(f"Unsupported environment type: {env_type}. Supported types: {list(_ENV_MODULES.keys())}")

    module = importlib.import_module(_ENV_MODULES[env_type], __name__)
    env_cls = getattr(module, env_type)
    _check_exec_tool_call_decorator(env_cls)
    globals()[env_type] = env_cls
    return env_cls


def __getattr__(name):
    if name in _ENV_MODULES:
        return _load_env_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_environment(env_type: str, max_turns: int, extra_info: dict = None) -> BaseEnv:
    """Factory function to create environment instances based on configuration.

    Args:
        env_type: Type of environment to create (e.g., 'MathSandboxEnv', 'CodeSandboxEnv')
        max_turns: Maximum number of turns for the environment
        extra_info: Extra information from dataset (e.g., prompt-dependent import code)

    Returns:
        BaseEnv instance

    Raises:
        ValueError: If env_type is not supported
    """
    env_cls = _load_env_class(env_type)
    return env_cls(max_turns=max_turns, extra_info=extra_info)


def _check_exec_tool_call_decorator(cls):
    """
    Check if a class's exec_tool_call method is decorated with with_timeout_and_retry.

    This function verifies that any override of exec_tool_call in a BaseEnv subclass
    has the required timeout and retry decorator applied.
    """
    if not issubclass(cls, BaseEnv):
        return

    # Skip BaseEnv itself
    if cls is BaseEnv:
        return

    # Check if the class overrides exec_tool_call
    if 'exec_tool_call' in cls.__dict__:
        method = cls.__dict__['exec_tool_call']

        # Check if the method is decorated by looking at its wrapper attributes
        # The with_timeout_and_retry decorator adds retry and timeout functionality
        if not (
            hasattr(method, '__wrapped__')
            or (hasattr(method, '__name__') and 'wrapper' in str(method))
            or (hasattr(method, '__code__') and method.__code__.co_name == 'wrapper')
        ):
            raise AssertionError(
                f"{cls.__name__}.exec_tool_call must be decorated with @with_timeout_and_retry. "
                f"Add the decorator like this:\n"
                f"@with_timeout_and_retry(timeout_seconds=30.0, max_attempts=3)\n"
                f"async def exec_tool_call(self, action: str) -> tuple[str, float, bool, dict]:\n"
                f"    ..."
            )


# Perform runtime check for BaseEnv itself without importing optional environments.
_check_exec_tool_call_decorator(BaseEnv)
