"""Adapter-state snapshots for PEFT training.

One live Megatron model and optimizer can time-share many adapter states when each
adapter's LoRA weights and optimizer state can be captured out of the live slot and
restored later. This module owns that optimizer-layout knowledge while staying
torch-only, so callers can unit-test adapter swapping without initializing Megatron.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

import torch

__all__ = [
    "AdapterState",
    "DistributedAdapterState",
    "MegatronMixedPrecisionAdapterState",
    "MixedPrecisionAdapterState",
    "flatten",
    "select_adapter_state",
    "unflatten_into",
]

_DISTRIBUTED_UNSUPPORTED = (
    "adapter-state snapshots do not support the distributed optimizer: it shards "
    "optimizer state across data-parallel ranks, so a per-adapter in-memory swap has "
    "no whole tensors to move. Set use_distributed_optimizer=False."
)


def flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Return tensors as one flat buffer in their current order."""
    return torch._utils._flatten_dense_tensors(tensors)


def unflatten_into(flat: torch.Tensor, dsts: list[torch.Tensor]) -> None:
    """Copy a flat buffer back into destination tensors in order."""
    for src, dst in zip(torch._utils._unflatten_dense_tensors(flat, dsts), dsts):
        dst.detach().copy_(src)


class AdapterState(ABC):
    """Stored adapter state for one run or tenant.

    Callers treat this as an opaque value. Implementations know how to capture and
    restore one optimizer layout.
    """

    version: int
    step: int

    @classmethod
    @abstractmethod
    def assert_supported(cls, optimizer: Any) -> None:
        """Raise if this optimizer cannot be captured and restored."""

    @classmethod
    @abstractmethod
    def snapshot_master(cls, optimizer: Any) -> torch.Tensor:
        """Return a flat copy of the fp32 master weights."""

    @classmethod
    @abstractmethod
    def genesis(cls, optimizer: Any, pristine_master: torch.Tensor) -> AdapterState:
        """Build the initial adapter state for a fresh adapter."""

    @classmethod
    @abstractmethod
    def capture(cls, optimizer: Any, version: int) -> AdapterState:
        """Copy the live optimizer state into an adapter snapshot."""

    @classmethod
    @abstractmethod
    def reset_optimizer_state(cls, optimizer: Any) -> None:
        """Make fp32 masters match model weights and clear optimizer history."""

    @classmethod
    @abstractmethod
    def trainable_params(cls, optimizer: Any) -> list[torch.Tensor]:
        """Return trainable model parameters in gradient-buffer order."""

    @abstractmethod
    def restore_into(self, optimizer: Any) -> None:
        """Copy this snapshot into the live optimizer and model tensors."""

    @abstractmethod
    def clone(self) -> AdapterState:
        """Return an independent copy of this adapter state."""

    @property
    @abstractmethod
    def nbytes(self) -> int:
        """Bytes held by this stored adapter state on this rank."""


@dataclass
class _MixedBinding:
    """References to tensors currently installed in the live optimizer."""

    model_params: list[torch.Tensor]
    master_params: list[torch.Tensor]
    opt_state: dict[str, list[torch.Tensor]]
    pgroups: list[dict]


@dataclass
class MegatronMixedPrecisionAdapterState(AdapterState):
    """Adapter snapshot for a non-distributed mixed-precision Megatron optimizer.

    Stored LoRA weights are fp32 master weights. Restore copies them into the
    optimizer masters, casts them back to the lower-precision model parameters, and
    restores tensor-valued optimizer state by key. This keeps the logic generic over
    Adam moments, SGD momentum buffers, and similar optimizer layouts.
    """

    lora: torch.Tensor
    opt_state: dict[str, torch.Tensor]
    step: int
    version: int

    @classmethod
    def assert_supported(cls, optimizer: Any) -> None:
        if not any(True for _ in cls._iter_master_pairs(optimizer)):
            raise NotImplementedError(
                "adapter-state snapshots require a half-precision (fp16/bf16) model "
                "with fp32 optimizer masters; this optimizer exposes no fp32 master "
                "params."
            )

    @classmethod
    def snapshot_master(cls, optimizer: Any) -> torch.Tensor:
        masters = [
            master for _model, master, _inner in cls._iter_master_pairs(optimizer)
        ]
        return flatten(masters).detach().clone()

    @classmethod
    def genesis(
        cls, optimizer: Any, pristine_master: torch.Tensor
    ) -> MegatronMixedPrecisionAdapterState:
        binding = cls._bind(optimizer)
        return cls(
            lora=pristine_master,
            opt_state={
                key: torch.zeros_like(flatten(tensors))
                for key, tensors in binding.opt_state.items()
            },
            step=0,
            version=0,
        )

    @classmethod
    def capture(
        cls, optimizer: Any, version: int
    ) -> MegatronMixedPrecisionAdapterState:
        binding = cls._bind(optimizer)
        return cls(
            lora=flatten(binding.master_params).detach().clone(),
            opt_state={
                key: flatten(tensors).detach().clone()
                for key, tensors in binding.opt_state.items()
            },
            step=cls._get_step(binding.pgroups),
            version=version,
        )

    @classmethod
    def reset_optimizer_state(cls, optimizer: Any) -> None:
        binding = cls._bind(optimizer)
        for model_param, master_param in zip(
            binding.model_params, binding.master_params
        ):
            master_param.data.copy_(model_param.data.float())
        for tensors in binding.opt_state.values():
            for tensor in tensors:
                tensor.zero_()
        cls._set_step(binding.pgroups, 0)

    @classmethod
    def trainable_params(cls, optimizer: Any) -> list[torch.Tensor]:
        return cls._bind(optimizer).model_params

    def restore_into(self, optimizer: Any) -> None:
        binding = type(self)._bind(optimizer)
        unflatten_into(self.lora, binding.master_params)
        for model_param, master_param in zip(
            binding.model_params, binding.master_params
        ):
            model_param.data.copy_(master_param.data.to(model_param.dtype))
        for key, tensors in binding.opt_state.items():
            unflatten_into(self.opt_state[key], tensors)
        type(self)._set_step(binding.pgroups, self.step)

    def clone(self) -> MegatronMixedPrecisionAdapterState:
        return MegatronMixedPrecisionAdapterState(
            lora=self.lora.clone(),
            opt_state={key: tensor.clone() for key, tensor in self.opt_state.items()},
            step=self.step,
            version=self.version,
        )

    @property
    def nbytes(self) -> int:
        tensors = [self.lora, *self.opt_state.values()]
        return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)

    @staticmethod
    def _children(optimizer: Any) -> list:
        """Return a ChainedOptimizer's children, or the optimizer itself."""
        return getattr(optimizer, "chained_optimizers", None) or [optimizer]

    @classmethod
    def _iter_master_pairs(
        cls, optimizer: Any
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, Any]]:
        """Yield ``(model_param, master_param, inner_optimizer)`` per trainable param.

        This is safe before the first optimizer step because fp32 masters already
        exist. Use ``_bind`` only after optimizer state tensors have been created.
        """
        for child in cls._children(optimizer):
            inner = getattr(child, "optimizer", None) or child
            model_groups = getattr(child, "float16_groups", None)
            master_groups = getattr(child, "fp32_from_float16_groups", None)
            if not (model_groups and master_groups):
                continue
            for model_group, master_group in zip(model_groups, master_groups):
                for model_param, master_param in zip(model_group, master_group):
                    yield model_param, master_param, inner

    @classmethod
    def _bind(cls, optimizer: Any) -> _MixedBinding:
        """Collect the live tensors needed to capture or restore an adapter."""
        model_params: list[torch.Tensor] = []
        master_params: list[torch.Tensor] = []
        opt_state: dict[str, list[torch.Tensor]] = {}
        state_keys: list[str] | None = None
        for model_param, master_param, inner in cls._iter_master_pairs(optimizer):
            state = inner.state.get(master_param, {})
            keys = [
                key
                for key, value in state.items()
                if key != "step" and isinstance(value, torch.Tensor)
            ]
            if not keys:
                continue
            if state_keys is None:
                state_keys = keys
                opt_state = {key: [] for key in state_keys}
            elif set(keys) != set(state_keys):
                raise RuntimeError(
                    f"optimizer state keys differ across params: {keys} vs "
                    f"{state_keys}; the adapter swap needs a uniform layout"
                )
            model_params.append(model_param)
            master_params.append(master_param)
            for key in state_keys:
                opt_state[key].append(state[key])
        pgroups: list[dict] = []
        for child in cls._children(optimizer):
            inner = getattr(child, "optimizer", None) or child
            pgroups.extend(inner.param_groups)
        if not master_params:
            raise RuntimeError("no fp32 master LoRA params found on the optimizer")
        return _MixedBinding(model_params, master_params, opt_state, pgroups)

    @staticmethod
    def _get_step(pgroups: list[dict]) -> int:
        """Read the first optimizer step counter found in the param groups."""
        for group in pgroups:
            if "step" in group:
                step = group["step"]
                return int(step.item()) if hasattr(step, "item") else int(step)
        return 0

    @staticmethod
    def _set_step(pgroups: list[dict], step: int) -> None:
        """Write the optimizer step counter to every param group that has one."""
        for group in pgroups:
            if "step" not in group:
                continue
            current = group["step"]
            if hasattr(current, "fill_"):
                current.fill_(step)
            else:
                group["step"] = step


MixedPrecisionAdapterState = MegatronMixedPrecisionAdapterState


class DistributedAdapterState(AdapterState):
    """Placeholder used to reject distributed optimizers with a clear error."""

    @classmethod
    def assert_supported(cls, optimizer: Any) -> None:
        raise NotImplementedError(_DISTRIBUTED_UNSUPPORTED)

    @classmethod
    def snapshot_master(cls, optimizer: Any) -> torch.Tensor:
        raise NotImplementedError(_DISTRIBUTED_UNSUPPORTED)

    @classmethod
    def genesis(cls, optimizer: Any, pristine_master: torch.Tensor) -> AdapterState:
        raise NotImplementedError(_DISTRIBUTED_UNSUPPORTED)

    @classmethod
    def capture(cls, optimizer: Any, version: int) -> AdapterState:
        raise NotImplementedError(_DISTRIBUTED_UNSUPPORTED)

    @classmethod
    def reset_optimizer_state(cls, optimizer: Any) -> None:
        raise NotImplementedError(_DISTRIBUTED_UNSUPPORTED)

    @classmethod
    def trainable_params(cls, optimizer: Any) -> list[torch.Tensor]:
        raise NotImplementedError(_DISTRIBUTED_UNSUPPORTED)

    def restore_into(self, optimizer: Any) -> None:
        raise NotImplementedError(_DISTRIBUTED_UNSUPPORTED)

    def clone(self) -> AdapterState:
        raise NotImplementedError(_DISTRIBUTED_UNSUPPORTED)

    @property
    def nbytes(self) -> int:
        raise NotImplementedError(_DISTRIBUTED_UNSUPPORTED)


def select_adapter_state(optimizer: Any, *, is_distributed: bool) -> type[AdapterState]:
    """Return the adapter-state implementation for this optimizer setup."""
    if is_distributed:
        return DistributedAdapterState
    return MegatronMixedPrecisionAdapterState
