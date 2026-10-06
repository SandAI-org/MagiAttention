# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
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

"""Out-of-process watchdog for the deterministic q/k-range kernels.

Each job runs in a child process on its own GPU; the parent only reads the
child's stdout and enforces wall-clock budgets, so a hung kernel cannot hang
the parent (it never touches the child's GPU state). A job is one direction
(fwd: 1-CTA O/LSE chain; bwd: 2-CTA dQ and reduced GQA dK/dV chains) and one
:class:`DelayPoint` (or none); it runs, in order, a single cluster, two
clusters and the full grid, the first and last also under a concurrent
matmul load on another stream. Every result must equal, bit for bit, the
undelayed deterministic result on the full grid.

Child protocol, one line each:
  ``MANIFEST <json>``  seed, config, commit and delay point, before any launch;
  ``WARM <case> <seconds>``  compiled, warmed up and checked; the seconds are
      one launch of the case, which sets the budget of its timed loop;
  ``PASS <case>``  every timed launch completed and matched;
  ``DONE``.
Budgets: compile phases ``_COMPILE_BUDGET_S``; a timed loop twice its
measured duration plus ``_SLACK_S``. A job exceeding a budget is killed and
its case rerun alone with 4x budgets; the result records the outcome
("first_timeout_rerun_passed" or "repeated_timeout") and fails the test
either way. Neither outcome alone tells a slow case from a timing-dependent
hang; the cause is analysed separately.
"""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from unittest import TestCase, mock

import pytest
from torch.testing._internal.common_utils import run_tests

_REPO = Path(__file__).resolve().parents[3]
_COMPILE_BUDGET_S = 900.0
_SLACK_S = 30.0
_ITERS = 20
_SEED = 20260718
# (case, clusters of the budget, None for the full grid; concurrent load)
_CASES: list[tuple[str, int | None, bool]] = [
    ("one_cluster", 1, False),
    ("one_cluster_loaded", 1, True),
    ("two_clusters", 2, False),
    ("full_grid", None, False),
    ("full_grid_loaded", None, True),
]
_DELAYS = [None, "start", "claim", "peer_arrive", "publish", "scan"]


# ---------------------------------------------------------------- child side


def _child(config: dict) -> None:
    import torch

    from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP
    from magi_attention.kernel.cutedsl.flex_flash_attn import (
        _flex_flash_attn_bwd,
        _flex_flash_attn_fwd,
    )
    from tests.test_kernel.cutedsl.range_deterministic_probes import (
        DelayPoint,
        delay_injection,
        fresh_compile_caches,
    )
    from tests.test_kernel.cutedsl.test_ffa_range_deterministic import (
        _overlapping_relations,
        _qkv,
        _range_args,
    )

    manifest = dict(config, device=torch.cuda.get_device_name(0))
    print("MANIFEST " + json.dumps(manifest), flush=True)
    torch.manual_seed(config["seed"])
    num_sm = torch.cuda.get_device_properties(0).multi_processor_count
    fwd = config["direction"] == "fwd"
    cluster = 1 if fwd else 2

    if fwd:
        relations, total_q, total_k = _overlapping_relations(
            [700, 1500, 300], coverage=4, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
    else:
        # Relations sharing K rows (A among themselves, B among themselves)
        # with overlapping Q rows (each B over two A), no repeated (q, k).
        relations = []
        for i in range(6):
            relations.append(
                ([5 + 600 * i, 525 + 600 * i], [7 + 60 * i, 807 + 60 * i], MT_MAP.full)
            )
            relations.append(
                (
                    [205 + 600 * i, 705 + 600 * i],
                    [2000 + 29 * i, 2600 + 29 * i],
                    MT_MAP.causal if i % 2 else MT_MAP.full,
                )
            )
        total_q, total_k = 3710, 2750
    args = _range_args(relations)
    q, k, v = _qkv(total_q, total_k, group=2)
    do = torch.randn_like(q)
    out_lse = _flex_flash_attn_fwd(
        q, k, v, **args, disable_fwd_atomic_reduction=False, deterministic=True
    )
    out, lse = out_lse[0].to(q.dtype), out_lse[1]

    def run(sm_margin: int, deterministic: bool = True):
        if fwd:
            return _flex_flash_attn_fwd(
                q,
                k,
                v,
                **args,
                disable_fwd_atomic_reduction=False,
                deterministic=deterministic,
                sm_margin=sm_margin,
            )
        return _flex_flash_attn_bwd(
            q,
            k,
            v,
            out,
            lse,
            do,
            **args,
            disable_bwd_dkv_atomic_reduction=False,
            deterministic=deterministic,
            sm_margin=sm_margin,
        )[:3]

    def timed(fn) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - start

    reference = [t.clone() for t in run(0)]
    run(0, deterministic=False)
    nondet_s = timed(lambda: run(0, deterministic=False))
    print(f"INFO nondeterministic full grid {nondet_s:.6f}s", flush=True)
    load_a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()

    def check(result, case: str) -> None:
        for i, (got, want) in enumerate(zip(result, reference)):
            if not torch.equal(got, want):
                raise AssertionError(f"{case}: output {i} differs from the reference")

    point = None if config["delay"] is None else DelayPoint(config["delay"])
    probe = (
        fresh_compile_caches()
        if point is None
        else delay_injection(point, seed=config["seed"])
    )
    with probe:
        for case, clusters, load in _CASES:
            if config["only"] not in (None, case):
                continue
            sm_margin = 0 if clusters is None else num_sm - clusters * cluster

            def launch():
                if load:
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(4):
                            load_a @ load_a
                return run(sm_margin)

            check(launch(), case)
            seconds = timed(launch)
            print(f"WARM {case} {seconds:.6f}", flush=True)
            results = [launch() for _ in range(config["iters"])]
            torch.cuda.synchronize()
            for result in results:
                check(result, case)
            print(f"PASS {case}", flush=True)
    print("DONE", flush=True)


# --------------------------------------------------------------- parent side


@dataclass
class _JobResult:
    job: str
    # "pass", "error", "first_timeout_rerun_passed" or "repeated_timeout";
    # the timeouts say only how the budgets were exceeded, not why.
    status: str
    detail: str


@dataclass
class _WatchResult:
    # "pass", "error" or "timeout" (``case`` names the case whose budget
    # ran out)
    status: str
    case: str
    log: str
    pid: int


def _watch(
    cmd: list[str],
    env: dict[str, str],
    iters: int,
    budget_scale: float,
    compile_budget_s: float = _COMPILE_BUDGET_S,
) -> _WatchResult:
    """Run one child under the budgets of the module docstring.

    Whatever ends the watch (the child exits, a budget runs out, a malformed
    protocol line, or an exception in the parent), the child's process group
    is killed if still running and the child is reaped before returning.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=_REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    log: list[str] = []
    case = "startup"
    try:
        threading.Thread(target=pump, daemon=True).start()
        deadline = time.monotonic() + compile_budget_s * budget_scale
        while True:
            try:
                line = lines.get(timeout=max(deadline - time.monotonic(), 0.0))
            except queue.Empty:
                return _WatchResult("timeout", case, "".join(log), proc.pid)
            if line is None:
                break
            log.append(line)
            parts = line.split()
            if not parts:
                continue
            word, *rest = parts
            if word == "WARM":
                try:
                    case, seconds = rest[0], float(rest[1])
                except (IndexError, ValueError):
                    log.append(f"[watchdog] malformed protocol line: {line!r}\n")
                    return _WatchResult("error", case, "".join(log), proc.pid)
                budget = 2 * iters * seconds + _SLACK_S
            elif word == "PASS":
                budget = compile_budget_s
            elif word == "DONE":
                budget = _SLACK_S
            else:
                continue
            deadline = time.monotonic() + budget * budget_scale
        proc.wait()
        passed = proc.returncode == 0 and bool(log) and log[-1] == "DONE\n"
        return _WatchResult("pass" if passed else "error", case, "".join(log), proc.pid)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _child_cmd(config: dict) -> list[str]:
    return [sys.executable, "-u", __file__, "--child", json.dumps(config)]


def _child_env(gpu: str) -> dict[str, str]:
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        PYTHONPATH=os.pathsep.join([str(_REPO), os.environ.get("PYTHONPATH", "")]),
        MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA="0",
    )


def _run_job(direction: str, delay: str | None, gpus: "queue.Queue[str]") -> _JobResult:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True
    ).stdout.strip()
    config = dict(
        direction=direction,
        delay=delay,
        seed=_SEED,
        iters=_ITERS,
        commit=commit,
        delay_method="trace-time patch, range_deterministic_probes.delay_injection",
        only=None,
    )
    name = f"{direction}/{delay or 'none'}"
    gpu = gpus.get()
    try:
        first = _watch(_child_cmd(config), _child_env(gpu), _ITERS, budget_scale=1.0)
        if first.status != "timeout":
            return _JobResult(name, first.status, first.log[-4000:])
        rerun = _watch(
            _child_cmd(dict(config, only=first.case)),
            _child_env(gpu),
            _ITERS,
            budget_scale=4.0,
        )
        status = (
            "first_timeout_rerun_passed"
            if rerun.status == "pass"
            else "repeated_timeout" if rerun.status == "timeout" else "error"
        )
        return _JobResult(
            name,
            status,
            f"case {first.case} exceeded its budget; rerun alone with 4x budgets: "
            f"{rerun.status}\n--- first run ---\n{first.log[-3000:]}"
            f"\n--- rerun ---\n{rerun.log[-3000:]}",
        )
    finally:
        gpus.put(gpu)


class TestWatchdogParent(TestCase):
    """The parent side alone, on fake children; no GPU."""

    def _fake_child(self, script: str) -> list[str]:
        return [sys.executable, "-u", "-c", script]

    def _assert_reaped(self, pid: int) -> None:
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_blank_and_free_form_lines_do_not_stop_the_timeout(self):
        """Blank lines and logs without the protocol keywords are skipped; a
        child that then hangs is killed at the budget and reaped."""
        result = _watch(
            self._fake_child(
                "import time\n"
                "print()\nprint('   ')\nprint('INFO some log')\n"
                "print('Traceback-like text: WARMish')\n"
                "time.sleep(600)\n"
            ),
            dict(os.environ),
            iters=1,
            budget_scale=1.0,
            compile_budget_s=3.0,
        )
        self.assertEqual(result.status, "timeout")
        self.assertIn("INFO some log", result.log)
        self._assert_reaped(result.pid)

    def test_malformed_protocol_line_is_an_error_and_kills_the_child(self):
        result = _watch(
            self._fake_child(
                "import time\nprint('WARM one_cluster not-a-number')\n"
                "time.sleep(600)\n"
            ),
            dict(os.environ),
            iters=1,
            budget_scale=1.0,
            compile_budget_s=60.0,
        )
        self.assertEqual(result.status, "error")
        self.assertIn("malformed protocol line", result.log)
        self._assert_reaped(result.pid)

    def test_parent_exception_still_kills_the_child(self):
        """An exception raised while the parent waits on the child's output
        propagates after the child is killed and reaped."""
        pids: list[int] = []
        popen = subprocess.Popen

        def recording_popen(*args, **kwargs):
            proc = popen(*args, **kwargs)
            pids.append(proc.pid)
            return proc

        def failing_get(self, *args, **kwargs):
            raise RuntimeError("parent failure")

        with mock.patch.object(subprocess, "Popen", recording_popen), mock.patch.object(
            queue.Queue, "get", failing_get
        ), self.assertRaisesRegex(RuntimeError, "parent failure"):
            _watch(
                self._fake_child("import time\ntime.sleep(600)\n"),
                dict(os.environ),
                iters=1,
                budget_scale=1.0,
            )
        (pid,) = pids
        self._assert_reaped(pid)

    def test_clean_exit_after_done_passes(self):
        result = _watch(
            self._fake_child(
                "print('MANIFEST {}')\nprint('WARM a 0.001')\nprint('PASS a')\n"
                "print('DONE')\n"
            ),
            dict(os.environ),
            iters=1,
            budget_scale=1.0,
        )
        self.assertEqual(result.status, "pass")


@pytest.mark.slow
class TestFfaRangeDeterministicStress(TestCase):
    def test_delayed_protocol_steps_complete_and_stay_bitwise(self):
        import torch

        from magi_attention.kernel.cutedsl.ffa_utils import get_device_arch

        if get_device_arch()[1] not in (10, 11):
            self.skipTest("deterministic q/k ranges require SM100/SM110")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_ids = (
            visible.split(",")
            if visible
            else [str(i) for i in range(torch.cuda.device_count())]
        )
        gpus: queue.Queue[str] = queue.Queue()
        for gpu in gpu_ids:
            gpus.put(gpu)
        jobs = [(d, p) for d in ("fwd", "bwd") for p in _DELAYS]
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            results = list(pool.map(lambda job: _run_job(*job, gpus), jobs))
        for result in results:
            print(f"{result.job}: {result.status}", flush=True)
            for line in result.detail.splitlines():
                if line.split(" ", 1)[0] in ("MANIFEST", "INFO", "WARM"):
                    print(f"  {line}", flush=True)
        failed = [r for r in results if r.status != "pass"]
        self.assertFalse(
            failed,
            "\n\n".join(f"{r.job}: {r.status}\n{r.detail}" for r in failed),
        )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        _child(json.loads(sys.argv[2]))
    else:
        run_tests()
