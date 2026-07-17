# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Tests for ``schedule_forked_save``.

``sorted`` of a large list holds the GIL at C level, a deterministic
stand-in for torch_dist serialization. ``torch.distributed.barrier`` is
patched out because these tests do not initialize a process group.
"""

from __future__ import annotations

import os

# macOS dev only; harmless elsewhere.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import threading
import time
from unittest.mock import patch

import pytest
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest

from megatron.bridge.training.checkpointing import schedule_forked_save


requires_fork = pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="schedule_forked_save uses os.fork (POSIX only)",
)


_GIL_HOLD_N = 4_000_000
_GIL_HOLD_DATA = [(i * 2654435761) & 0xFFFFFFFF for i in range(_GIL_HOLD_N)]


def _gil_holding_write(marker_path: str) -> None:
    sorted(_GIL_HOLD_DATA)
    with open(marker_path, "w") as fh:
        fh.write(str(os.getpid()))


class _HealthProbe(threading.Thread):
    """Records the max wake-up gap; balloons when a peer thread holds the GIL."""

    def __init__(self, period: float = 0.01) -> None:
        super().__init__(daemon=True)
        self.period = period
        self._stop_event = threading.Event()
        self.ticks = 0
        self.max_gap = 0.0

    def run(self) -> None:
        last = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            gap = now - last
            last = now
            self.ticks += 1
            self.max_gap = max(self.max_gap, gap)
            time.sleep(self.period)

    def stop(self) -> None:
        self._stop_event.set()


@requires_fork
def test_forked_finalize_keeps_probe_responsive(tmp_path):
    marker = tmp_path / "marker.pid"
    request = AsyncRequest(
        async_fn=_gil_holding_write,
        async_fn_args=(str(marker),),
        finalize_fns=[],
    )

    probe = _HealthProbe(period=0.01)
    probe.start()
    time.sleep(0.05)
    with patch("megatron.bridge.training.checkpointing.torch.distributed.barrier"):
        schedule_forked_save(global_state=None, async_request=request)
    probe.stop()
    probe.join(timeout=5)

    assert marker.is_file()
    assert int(marker.read_text()) != os.getpid()
    assert probe.max_gap < 0.1, f"parent stalled {probe.max_gap:.3f}s"
    assert probe.ticks >= 20


@requires_fork
def test_forked_finalize_runs_finalize_fns_in_parent(tmp_path):
    marker = tmp_path / "marker.pid"
    finalize_pids: list[int] = []
    request = AsyncRequest(
        async_fn=_gil_holding_write,
        async_fn_args=(str(marker),),
        finalize_fns=[lambda: finalize_pids.append(os.getpid())],
    )

    with patch("megatron.bridge.training.checkpointing.torch.distributed.barrier"):
        schedule_forked_save(global_state=None, async_request=request)

    assert marker.is_file()
    assert int(marker.read_text()) != os.getpid()
    assert finalize_pids == [os.getpid()]


@requires_fork
def test_forked_finalize_raises_when_child_write_fails():
    def _boom(*_args, **_kwargs) -> None:
        raise ValueError("simulated write failure")

    finalize_pids: list[int] = []
    request = AsyncRequest(
        async_fn=_boom,
        async_fn_args=(),
        finalize_fns=[lambda: finalize_pids.append(os.getpid())],
    )

    with patch("megatron.bridge.training.checkpointing.torch.distributed.barrier"):
        with pytest.raises(RuntimeError, match="forked checkpoint finalize failed"):
            schedule_forked_save(global_state=None, async_request=request)

    assert finalize_pids == []


@requires_fork
def test_forked_finalize_child_can_log(tmp_path):
    import logging as _logging

    marker = tmp_path / "logged.pid"

    def _child_logs(path: str) -> None:
        _logging.getLogger("schedule_forked_save.test").info("child pid=%d", os.getpid())
        with open(path, "w") as fh:
            fh.write(str(os.getpid()))

    request = AsyncRequest(
        async_fn=_child_logs,
        async_fn_args=(str(marker),),
        finalize_fns=[],
    )

    with patch("megatron.bridge.training.checkpointing.torch.distributed.barrier"):
        schedule_forked_save(global_state=None, async_request=request)

    assert marker.is_file()
    assert int(marker.read_text()) != os.getpid()


def test_forked_finalize_falls_back_when_fork_unavailable(tmp_path, monkeypatch):
    marker = tmp_path / "marker.pid"
    finalize_pids: list[int] = []

    def _fast_write(path: str) -> None:
        with open(path, "w") as fh:
            fh.write(str(os.getpid()))

    request = AsyncRequest(
        async_fn=_fast_write,
        async_fn_args=(str(marker),),
        finalize_fns=[lambda: finalize_pids.append(os.getpid())],
    )
    monkeypatch.delattr(os, "fork", raising=True)

    with patch("megatron.bridge.training.checkpointing.torch.distributed.barrier"):
        schedule_forked_save(global_state=None, async_request=request)

    assert marker.is_file()
    assert int(marker.read_text()) == os.getpid()
    assert finalize_pids == [os.getpid()]
