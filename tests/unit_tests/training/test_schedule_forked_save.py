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
"""Tests that ``schedule_forked_save`` keeps the parent process responsive
during the (GIL-holding) async-checkpoint write by forking it, so a
co-resident FastAPI event loop / kubelet liveness probe is not starved and
SIGTERMed mid-write.

The production GIL hog is the per-rank torch_dist write (Python + C-level
serialization) invoked from ``AsyncRequest.async_fn``. When the writer
shares a process with a server event loop, that GIL hold freezes
``/health`` and trips the liveness probe, which SIGTERMs the pod
mid-save. ``schedule_forked_save`` forks the write into a child so the
parent blocks in ``os.waitpid`` — a GIL-releasing syscall — and the event
loop keeps running; it falls back to inline ``execute_sync`` semantics
only where ``os.fork`` is unavailable.

The GIL hold is simulated deterministically here (``sorted`` of a large
list does not release the GIL), so the timing assertions do not depend on
machine-specific serialization speed. A ``_HealthProbe`` thread plays the
liveness probe. ``torch.distributed.barrier`` is patched out because
these tests do not initialize a process group; the barrier's real
behavior is exercised by the full checkpointing integration tests.
"""

from __future__ import annotations

import os

# Allow fork() in a threaded process on macOS dev machines (CI is Linux). Must
# be set before the first fork; harmless elsewhere.
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


# A pre-built, well-shuffled list. ``sorted`` of it is a single C call that
# holds the GIL for ~0.7s (calibrated) — a deterministic stand-in for
# torch_dist serialization. Built once at import (in the parent); a forked
# child inherits it copy-on-write.
_GIL_HOLD_N = 4_000_000
_GIL_HOLD_DATA = [(i * 2654435761) & 0xFFFFFFFF for i in range(_GIL_HOLD_N)]


def _gil_holding_write(marker_path: str) -> None:
    """Stand-in for the real ``async_fn``: hold the GIL, record which PID ran it."""
    sorted(_GIL_HOLD_DATA)  # holds the GIL for its whole duration
    with open(marker_path, "w") as fh:
        fh.write(str(os.getpid()))


class _HealthProbe(threading.Thread):
    """Simulates the kubelet liveness probe / event-loop heartbeat.

    Wakes every ``period`` and records the realized gap between wake-ups. The
    gap balloons whenever another thread holds the GIL (this thread cannot
    reacquire it to run) — the starvation that freezes ``/health``.
    """

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


def _make_gil_holding_request(marker_path: str) -> AsyncRequest:
    return AsyncRequest(
        async_fn=_gil_holding_write,
        async_fn_args=(marker_path,),
        finalize_fns=[],
    )


@requires_fork
def test_forked_finalize_keeps_probe_responsive(tmp_path):
    """The parent stays responsive while the (GIL-holding) write runs in the
    forked child: the health probe keeps ticking near real-time, and the
    marker file records a child PID (not the parent's)."""
    marker = tmp_path / "marker.pid"
    request = _make_gil_holding_request(str(marker))

    probe = _HealthProbe(period=0.01)
    probe.start()
    time.sleep(0.05)  # let the probe establish a steady cadence
    with patch("megatron.bridge.training.checkpointing.torch.distributed.barrier"):
        schedule_forked_save(global_state=None, async_request=request)
    probe.stop()
    probe.join(timeout=5)

    # The child did the work (marker carries a different pid than ours).
    assert marker.is_file()
    assert int(marker.read_text()) != os.getpid()
    # Parent never starved: max gap stayed near the 10ms period, far below the
    # ~0.7s GIL hold, and the probe ticked many times across the write.
    assert probe.max_gap < 0.1, (
        f"parent stalled {probe.max_gap:.3f}s during forked save"
    )
    assert probe.ticks >= 20, f"probe only ticked {probe.ticks}x"


@requires_fork
def test_forked_finalize_runs_finalize_fns_in_parent(tmp_path):
    """``finalize_fns`` run on the parent process after the child exits.

    This matters because the finalize callbacks in the real save path (version
    pointer bump, cleanup) need the parent's NCCL state and filesystem view.
    """
    marker = tmp_path / "marker.pid"
    finalize_pids: list[int] = []
    request = AsyncRequest(
        async_fn=_gil_holding_write,
        async_fn_args=(str(marker),),
        finalize_fns=[lambda: finalize_pids.append(os.getpid())],
    )

    with patch("megatron.bridge.training.checkpointing.torch.distributed.barrier"):
        schedule_forked_save(global_state=None, async_request=request)

    # Child wrote the marker with its own pid; finalize ran in the parent.
    assert marker.is_file()
    assert int(marker.read_text()) != os.getpid()
    assert finalize_pids == [os.getpid()]


@requires_fork
def test_forked_finalize_raises_when_child_write_fails():
    """A raise in ``async_fn`` inside the child surfaces as ``RuntimeError`` in
    the parent, with a non-zero waitpid status. The parent-side barrier +
    finalize_fns must not run when the child failed (they would either see a
    half-written checkpoint or run against inconsistent state)."""

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

    # Finalize must NOT have run — the parent bailed before the trailer.
    assert finalize_pids == []


def test_forked_finalize_falls_back_when_fork_unavailable(tmp_path, monkeypatch):
    """When ``os.fork`` is missing (Windows / restricted sandboxes) the
    function degrades to ``execute_sync``-style inline semantics: async_fn +
    finalize_fns all run in the calling process. The event loop is not
    preserved in this path (nothing can be), but the caller sees identical
    correctness."""
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

    # Everything ran in the caller — same PID for both.
    assert marker.is_file()
    assert int(marker.read_text()) == os.getpid()
    assert finalize_pids == [os.getpid()]
