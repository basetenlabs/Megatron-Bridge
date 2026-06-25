# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Thread-safe HuggingFace remote-code loading utilities.

This module provides utilities for safely loading HuggingFace model configurations
and trust_remote_code dynamic modules in multi-process environments, preventing
race conditions that can occur when multiple processes try to download and cache
the same model simultaneously.
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Type, Union

import filelock
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import megatron.bridge.models.conversion.transformers_compat  # noqa: F401  # patches removed HF utils


def _get_hf_model_lock_file(path: Union[str, Path]) -> Path:
    """Return the cross-process lock file path for a HuggingFace model path."""
    path_hash = hashlib.md5(str(path).encode()).hexdigest()
    lock_dir = os.getenv("MEGATRON_CONFIG_LOCK_DIR")
    if lock_dir:
        lock_file = Path(lock_dir) / f".megatron_config_lock_{path_hash}"
    else:
        lock_file = Path.home() / ".cache" / "huggingface" / f".megatron_config_lock_{path_hash}"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    return lock_file


def _is_permanent_hf_load_error(error: Exception) -> bool:
    error_msg = str(error).lower()
    return any(
        phrase in error_msg
        for phrase in [
            "does not appear to have a file named config.json",
            "repository not found",
            "entry not found",
            "401 client error",
            "403 client error",
        ]
    )


def safe_load_config_with_retry(
    path: Union[str, Path], trust_remote_code: bool = False, max_retries: int = 3, base_delay: float = 1.0, **kwargs
) -> PretrainedConfig:
    """
    Thread-safe and process-safe configuration loading with retry logic.

    This function prevents race conditions when multiple threads/processes
    try to download and cache the same model configuration simultaneously.
    Uses file locking (if filelock is available) to coordinate access across
    processes.

    Args:
        path: HuggingFace model ID or path to model directory
        trust_remote_code: Whether to trust remote code when loading config
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        **kwargs: Additional arguments passed to AutoConfig.from_pretrained

    Returns:
        PretrainedConfig: The loaded model configuration

    Raises:
        ValueError: If config loading fails after all retries

    Environment Variables:
        MEGATRON_CONFIG_LOCK_DIR: Override the directory where lock files are created.
            Default: ~/.cache/huggingface/
            Useful for multi-node setups where a shared lock directory is needed.

    Example:
        >>> config = safe_load_config_with_retry("meta-llama/Meta-Llama-3-8B")
        >>> print(config.model_type)

        >>> # With custom retry settings
        >>> config = safe_load_config_with_retry(
        ...     "gpt2",
        ...     max_retries=5,
        ...     base_delay=0.5,
        ...     trust_remote_code=True
        ... )

        >>> # Multi-node setup with shared lock directory
        >>> import os
        >>> os.environ["MEGATRON_CONFIG_LOCK_DIR"] = "/shared/locks"
        >>> config = safe_load_config_with_retry("meta-llama/Meta-Llama-3-8B")
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            lock_file = _get_hf_model_lock_file(path)
            with filelock.FileLock(str(lock_file) + ".lock", timeout=60):
                return AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code, **kwargs)

        except Exception as e:
            last_exception = e

            if _is_permanent_hf_load_error(e):
                # Model doesn't exist or access denied, no point retrying
                raise ValueError(
                    f"Failed to load configuration from {path}. "
                    f"Ensure the path is valid and contains a config.json file. "
                    f"Error: {e}"
                ) from e

            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2**attempt) + (time.time() % 1) * 0.1
                time.sleep(delay)
            else:
                # Final attempt failed
                break

    # All retries exhausted
    raise ValueError(
        f"Failed to load configuration from {path} after {max_retries + 1} attempts. "
        f"This might be due to network issues or concurrent access conflicts. "
        f"Last error: {last_exception}"
    ) from last_exception


def safe_load_class_from_dynamic_module(
    class_reference: str,
    pretrained_model_name_or_path: Union[str, Path],
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
) -> Type:
    """
    Thread-safe and process-safe dynamic module class loading with retry logic.

    This function prevents race conditions when multiple threads/processes
    concurrently populate the shared HuggingFace dynamic-modules cache via
    ``get_class_from_dynamic_module``. Uses the same file lock as
    :func:`safe_load_config_with_retry` for a given model path.

    Args:
        class_reference: Fully qualified class name (e.g. ``module.ClassName``)
        pretrained_model_name_or_path: HuggingFace model ID or local model directory
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        **kwargs: Additional arguments passed to ``get_class_from_dynamic_module``

    Returns:
        The loaded class object

    Raises:
        ValueError: If class loading fails after all retries

    Environment Variables:
        MEGATRON_CONFIG_LOCK_DIR: Override the directory where lock files are created.
            Default: ~/.cache/huggingface/
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            lock_file = _get_hf_model_lock_file(pretrained_model_name_or_path)
            with filelock.FileLock(str(lock_file) + ".lock", timeout=60):
                return get_class_from_dynamic_module(
                    class_reference,
                    pretrained_model_name_or_path,
                    **kwargs,
                )

        except Exception as e:
            last_exception = e

            if _is_permanent_hf_load_error(e):
                raise ValueError(
                    f"Failed to load class {class_reference!r} from {pretrained_model_name_or_path}. "
                    f"Ensure the path is valid and contains the required remote code. "
                    f"Error: {e}"
                ) from e

            if attempt < max_retries:
                delay = base_delay * (2**attempt) + (time.time() % 1) * 0.1
                time.sleep(delay)
            else:
                break

    raise ValueError(
        f"Failed to load class {class_reference!r} from {pretrained_model_name_or_path} "
        f"after {max_retries + 1} attempts. "
        f"This might be due to network issues or concurrent access conflicts. "
        f"Last error: {last_exception}"
    ) from last_exception
