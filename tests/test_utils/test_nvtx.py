# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import tempfile
import unittest
from unittest import TestCase, mock

import torch
import torch.distributed as dist

from magi_attention.utils import nvtx


class TestNsysController(TestCase):
    def test_only_local_rank_zero_controls_profiled_node(self):
        with (
            mock.patch.object(nvtx, "_get_local_rank", return_value=0),
            mock.patch.object(nvtx, "_get_local_world_size", return_value=8),
        ):
            self.assertTrue(nvtx._is_nsys_controller(8, [9]))
            self.assertFalse(nvtx._is_nsys_controller(8, [0]))

        with (
            mock.patch.object(nvtx, "_get_local_rank", return_value=1),
            mock.patch.object(nvtx, "_get_local_world_size", return_value=8),
        ):
            self.assertFalse(nvtx._is_nsys_controller(9, [9]))


class TestPhaseSelector(TestCase):
    def test_matches_comma_separated_exact_and_glob_selectors(self):
        selector = "init/ckpt_load, data/train_*"

        self.assertTrue(nvtx._matches_phase_selector("init/ckpt_load", selector))
        self.assertTrue(nvtx._matches_phase_selector("data/train_first_load", selector))
        self.assertFalse(nvtx._matches_phase_selector("data/val_first_load", selector))

    def test_matches_explicit_regular_expression_selector(self):
        selector = r"re:init/(model|ckpt_load),data/train_first_load"

        self.assertTrue(nvtx._matches_phase_selector("init/model", selector))
        self.assertTrue(nvtx._matches_phase_selector("init/ckpt_load", selector))
        self.assertFalse(nvtx._matches_phase_selector("init/data_loader", selector))


class TestProfilePhase(TestCase):
    def setUp(self):
        nvtx._ACTIVE_NSYS_PHASE = None

    def test_unselected_phase_only_logs_cpu_wall_time(self):
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(nvtx.logger, "info") as log_info,
            mock.patch.object(torch.cuda, "cudart") as cudart,
        ):
            with nvtx.profile_phase("init/model", category="cold_start"):
                pass

        cudart.assert_not_called()
        self.assertIn(
            "[PHASE][cold_start][cpu-wall] init/model:", log_info.call_args.args[0]
        )

    def test_selected_phase_starts_and_stops_nsys_capture(self):
        cudart = mock.Mock()
        nsys_command = (
            [
                "nsys",
                "profile",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=repeat:10:sync",
            ],
            "/workspace",
        )
        with (
            mock.patch.dict(
                "os.environ",
                {nvtx.NSYS_CAPTURE_PHASE_ENV: "init/*"},
                clear=True,
            ),
            mock.patch.object(nvtx, "_get_nsys_command", return_value=nsys_command),
            mock.patch.object(nvtx, "_is_nsys_controller", return_value=True),
            mock.patch.object(nvtx, "_snapshot_nsys_reports"),
            mock.patch.object(nvtx, "_get_node_rank", return_value=0),
            mock.patch.object(nvtx, "_rename_latest_nsys_report") as rename_report,
            mock.patch.object(dist, "is_initialized", return_value=False),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
            mock.patch.object(torch.cuda.nvtx, "range_push") as range_push,
            mock.patch.object(torch.cuda.nvtx, "range_pop") as range_pop,
        ):
            with nvtx.profile_phase("init/ckpt_load", category="cold_start"):
                pass

        cudart.cudaProfilerStart.assert_called_once_with()
        cudart.cudaProfilerStop.assert_called_once_with()
        range_push.assert_called_once_with("init/ckpt_load")
        range_pop.assert_called_once_with()
        rename_report.assert_called_once_with(
            "profile_phase(init_ckpt_load)_n0.nsys-rep"
        )

    def test_capture_can_be_disabled_for_matching_runtime_phase(self):
        with (
            mock.patch.dict(
                "os.environ",
                {nvtx.NSYS_CAPTURE_PHASE_ENV: "data/*"},
                clear=True,
            ),
            mock.patch.object(nvtx, "_get_nsys_command") as nsys_command,
        ):
            with nvtx.profile_phase(
                "data/val_first_load", category="runtime", capture_nsys=False
            ):
                pass

        nsys_command.assert_not_called()


class TestSwitchProfile(TestCase):
    def setUp(self):
        nvtx._PROFILER_ENABLED = False
        nvtx._EMIT_NVTX_CTX = None
        nvtx._NSYS_REPORT_FILES_BEFORE = {}
        nvtx._NSYS_REPORT_DIR = None
        nvtx._NSYS_REPORT_BASE = None
        nvtx._NSYS_COMMAND = None
        nvtx._NSYS_COMMAND_DISCOVERED = False
        nvtx._NSYS_CAPTURE_LIMIT_WARNED = True
        nsys_command_patch = mock.patch.object(
            nvtx,
            "_get_nsys_command",
            return_value=(
                ["nsys", "profile", "--capture-range-end=repeat:1000:sync"],
                "/workspace",
            ),
        )
        nsys_command_patch.start()
        self.addCleanup(nsys_command_patch.stop)

    def _distributed_mocks(self, rank: int, local_rank: int):
        cudart = mock.Mock()
        return (
            cudart,
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "get_rank", return_value=rank),
            mock.patch.object(dist, "barrier"),
            mock.patch.object(torch.cuda, "cudart", return_value=cudart),
            mock.patch.object(torch.cuda.nvtx, "range_push"),
            mock.patch.object(torch.cuda.nvtx, "range_pop"),
            mock.patch.object(nvtx, "_get_local_rank", return_value=local_rank),
            mock.patch.object(nvtx, "_get_local_world_size", return_value=8),
        )

    def test_disabled_profile_is_a_noop(self):
        with (
            mock.patch.object(dist, "is_initialized") as initialized,
            mock.patch.object(torch.cuda, "cudart") as cudart,
            mock.patch.object(torch.cuda.memory, "_record_memory_history") as history,
        ):
            nvtx.switch_profile(
                iter_id=2,
                start=2,
                end=4,
                profile_ranks=[0],
                profile_type=["nsys", "memory"],
                enable=False,
            )

        initialized.assert_not_called()
        cudart.assert_not_called()
        history.assert_not_called()
        self.assertFalse(nvtx._PROFILER_ENABLED)

    def test_profile_window_repeats_at_interval(self):
        self.assertEqual(nvtx._get_profile_window(2, 2, 4, 10), (2, 4))
        self.assertIsNone(nvtx._get_profile_window(5, 2, 4, 10))
        self.assertEqual(nvtx._get_profile_window(12, 2, 4, 10), (12, 14))
        self.assertEqual(nvtx._get_profile_window(14, 2, 4, 10), (12, 14))

    def test_profile_interval_rejects_overlapping_windows(self):
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            nvtx._get_profile_window(2, 2, 4, 2)

    def test_continual_profile_accepts_bounded_or_unbounded_sync_repeat(self):
        self.assertEqual(
            nvtx._validate_nsys_command(
                ["nsys", "profile", "--capture-range-end=repeat:2:sync"],
                exit_after_end_iter=False,
                interval=10,
                max_capture_ranges=2,
            ),
            2,
        )
        self.assertIsNone(
            nvtx._validate_nsys_command(
                ["nsys", "profile", "--capture-range-end", "repeat:sync"],
                exit_after_end_iter=False,
                interval=10,
                max_capture_ranges=1000,
            )
        )

    def test_continual_profile_rejects_non_sync_capture_end(self):
        with self.assertRaisesRegex(ValueError, "repeat:N:sync or repeat:sync"):
            nvtx._validate_nsys_command(
                ["nsys", "profile", "--capture-range-end=repeat:2"],
                exit_after_end_iter=False,
                interval=10,
                max_capture_ranges=2,
            )

    def test_nsys_repeat_count_must_cover_max_capture_ranges(self):
        with self.assertRaisesRegex(ValueError, "allows 2.*max_capture_ranges=3"):
            nvtx._validate_nsys_command(
                ["nsys", "profile", "--capture-range-end=repeat:2:sync"],
                exit_after_end_iter=False,
                interval=10,
                max_capture_ranges=3,
            )

    def test_exit_after_end_requires_one_shot_stop(self):
        self.assertEqual(
            nvtx._validate_nsys_command(
                ["nsys", "profile", "--capture-range-end=stop"],
                exit_after_end_iter=True,
                interval=0,
                max_capture_ranges=1000,
            ),
            1,
        )
        with self.assertRaisesRegex(ValueError, "interval must be 0"):
            nvtx._validate_nsys_command(
                ["nsys", "profile", "--capture-range-end=stop"],
                exit_after_end_iter=True,
                interval=10,
                max_capture_ranges=1000,
            )
        with self.assertRaisesRegex(ValueError, "requires.*stop"):
            nvtx._validate_nsys_command(
                ["nsys", "profile", "--capture-range-end=repeat:sync"],
                exit_after_end_iter=True,
                interval=0,
                max_capture_ranges=1000,
            )

    def test_missing_nsys_wrapper_disables_all_profiling(self):
        with (
            mock.patch.object(nvtx, "_get_nsys_command", return_value=None),
            mock.patch.object(dist, "is_initialized") as initialized,
            mock.patch.object(torch.cuda, "cudart") as cudart,
            mock.patch.object(torch.cuda.memory, "_record_memory_history") as history,
        ):
            nvtx.switch_profile(
                iter_id=2,
                start=2,
                end=4,
                profile_ranks=[0],
                profile_type=["nsys", "memory"],
            )

        initialized.assert_not_called()
        cudart.assert_not_called()
        history.assert_not_called()

    def test_nsys_output_dir_is_read_from_wrapper_command(self):
        self.assertEqual(
            nvtx._get_nsys_output_base(
                ["/opt/nsys", "profile", "-o", "reports/profile_n0"],
                "/workspace",
            ),
            "/workspace/reports/profile_n0",
        )
        self.assertEqual(
            nvtx._get_nsys_output_dir(
                ["/opt/nsys", "profile", "-o", "reports/profile_n0"],
                "/workspace",
            ),
            "/workspace/reports",
        )
        self.assertEqual(
            nvtx._get_nsys_output_dir(
                ["nsys", "profile", "--output=/logs/profile_n0"],
                "/workspace",
            ),
            "/logs",
        )
        self.assertIsNone(
            nvtx._get_nsys_output_dir(["torchrun", "train.py"], "/workspace")
        )

    def test_nsys_report_is_renamed_with_window_and_node(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_base = os.path.join(tmp_dir, "profile_n3")
            old_report = f"{report_base}.1.nsys-rep"
            with open(old_report, "w", encoding="utf-8") as f:
                f.write("old")
            foreign_report = os.path.join(tmp_dir, "profile_n4.2.nsys-rep")
            prefix_collision_report = os.path.join(tmp_dir, "profile_n30.2.nsys-rep")
            nvtx._NSYS_COMMAND = (
                ["nsys", "profile", "-o", os.path.join(tmp_dir, "profile_n3")],
                "/workspace",
            )
            with mock.patch.object(nvtx, "_find_nsys_command") as find_command:
                nvtx._snapshot_nsys_reports()
                find_command.assert_not_called()
                new_report = f"{report_base}.2.nsys-rep"
                with open(new_report, "w", encoding="utf-8") as f:
                    f.write("new")
                with open(foreign_report, "w", encoding="utf-8") as f:
                    f.write("foreign")
                with open(prefix_collision_report, "w", encoding="utf-8") as f:
                    f.write("prefix collision")
                nvtx._rename_nsys_report(12, 14, node_rank=3)

            renamed_report = os.path.join(tmp_dir, "profile_iter(12-14)_n3.nsys-rep")
            self.assertTrue(os.path.exists(old_report))
            self.assertFalse(os.path.exists(new_report))
            self.assertTrue(os.path.exists(foreign_report))
            self.assertTrue(os.path.exists(prefix_collision_report))
            with open(renamed_report, encoding="utf-8") as f:
                self.assertEqual(f.read(), "new")

    def test_second_periodic_window_starts_and_stops(self):
        (
            cudart,
            initialized,
            get_rank,
            barrier,
            cudart_patch,
            push,
            pop,
            local_rank,
            local_world_size,
        ) = self._distributed_mocks(rank=0, local_rank=0)
        with (
            initialized,
            get_rank,
            barrier as barrier_mock,
            cudart_patch,
            push as push_mock,
            pop as pop_mock,
            local_rank,
            local_world_size,
            mock.patch.object(nvtx, "_snapshot_nsys_reports"),
            mock.patch.object(nvtx, "_rename_nsys_report"),
        ):
            nvtx.switch_profile(
                iter_id=12,
                start=2,
                end=4,
                interval=10,
                profile_ranks=[0],
                record_shape=False,
            )
            nvtx.switch_profile(
                iter_id=14,
                start=2,
                end=4,
                interval=10,
                profile_ranks=[0],
                record_shape=False,
            )

        cudart.cudaProfilerStart.assert_called_once_with()
        cudart.cudaProfilerStop.assert_called_once_with()
        push_mock.assert_called_once_with("iter_12")
        pop_mock.assert_called_once_with()
        self.assertEqual(barrier_mock.call_count, 3)
        self.assertFalse(nvtx._PROFILER_ENABLED)

    def test_capture_limit_makes_later_nsys_windows_noop(self):
        (
            cudart,
            initialized,
            get_rank,
            barrier,
            cudart_patch,
            push,
            pop,
            local_rank,
            local_world_size,
        ) = self._distributed_mocks(rank=0, local_rank=0)
        with (
            initialized,
            get_rank,
            barrier as barrier_mock,
            cudart_patch,
            push as push_mock,
            pop,
            local_rank,
            local_world_size,
        ):
            nvtx.switch_profile(
                iter_id=22,
                start=2,
                end=4,
                interval=10,
                max_capture_ranges=2,
                profile_ranks=[0],
                record_shape=False,
            )

        cudart.cudaProfilerStart.assert_not_called()
        barrier_mock.assert_not_called()
        push_mock.assert_not_called()
        self.assertFalse(nvtx._PROFILER_ENABLED)

    def test_exit_after_end_iter_leaves_process_lifecycle_to_nsys(self):
        (
            cudart,
            initialized,
            get_rank,
            barrier,
            cudart_patch,
            push,
            pop,
            local_rank,
            local_world_size,
        ) = self._distributed_mocks(rank=0, local_rank=0)
        with (
            initialized,
            get_rank,
            barrier as barrier_mock,
            cudart_patch,
            push,
            pop as pop_mock,
            local_rank,
            local_world_size,
            mock.patch.object(
                nvtx,
                "_get_nsys_command",
                return_value=(
                    ["nsys", "profile", "--capture-range-end=stop"],
                    "/workspace",
                ),
            ),
        ):
            nvtx.switch_profile(
                iter_id=4,
                start=2,
                end=4,
                profile_ranks=[0],
                record_shape=False,
                exit_after_end_iter=True,
            )

        pop_mock.assert_called_once_with()
        cudart.cudaProfilerStop.assert_called_once_with()
        self.assertEqual(barrier_mock.call_count, 2)

    def test_controller_can_be_separate_from_instrumented_rank(self):
        (
            cudart,
            initialized,
            get_rank,
            barrier,
            cudart_patch,
            push,
            pop,
            local_rank,
            local_world_size,
        ) = self._distributed_mocks(rank=0, local_rank=0)
        with (
            initialized,
            get_rank,
            barrier as barrier_mock,
            cudart_patch,
            push as push_mock,
            pop,
            local_rank,
            local_world_size,
        ):
            nvtx.switch_profile(
                iter_id=2,
                start=2,
                end=4,
                profile_ranks=[1],
                record_shape=False,
            )

        cudart.cudaProfilerStart.assert_called_once_with()
        barrier_mock.assert_called_once_with()
        push_mock.assert_not_called()
        self.assertFalse(nvtx._PROFILER_ENABLED)

    def test_controller_stops_session_when_it_is_not_instrumented(self):
        (
            cudart,
            initialized,
            get_rank,
            barrier,
            cudart_patch,
            push,
            pop,
            local_rank,
            local_world_size,
        ) = self._distributed_mocks(rank=0, local_rank=0)
        with (
            initialized,
            get_rank,
            barrier as barrier_mock,
            cudart_patch,
            push,
            pop as pop_mock,
            local_rank,
            local_world_size,
        ):
            nvtx.switch_profile(
                iter_id=4,
                start=2,
                end=4,
                profile_ranks=[1],
                record_shape=False,
            )

        pop_mock.assert_not_called()
        cudart.cudaProfilerStop.assert_called_once_with()
        self.assertEqual(barrier_mock.call_count, 2)

    def test_profile_rank_is_instrumented_without_controlling_session(self):
        (
            cudart,
            initialized,
            get_rank,
            barrier,
            cudart_patch,
            push,
            pop,
            local_rank,
            local_world_size,
        ) = self._distributed_mocks(rank=1, local_rank=1)
        with (
            initialized,
            get_rank,
            barrier as barrier_mock,
            cudart_patch,
            push as push_mock,
            pop,
            local_rank,
            local_world_size,
        ):
            nvtx.switch_profile(
                iter_id=2,
                start=2,
                end=4,
                profile_ranks=[1],
                record_shape=False,
            )

        cudart.cudaProfilerStart.assert_not_called()
        barrier_mock.assert_called_once_with()
        push_mock.assert_called_once_with("iter_2")
        self.assertTrue(nvtx._PROFILER_ENABLED)

    def test_stop_waits_for_synchronous_report_before_returning(self):
        (
            cudart,
            initialized,
            get_rank,
            barrier,
            cudart_patch,
            push,
            pop,
            local_rank,
            local_world_size,
        ) = self._distributed_mocks(rank=0, local_rank=0)
        with (
            initialized,
            get_rank,
            barrier as barrier_mock,
            cudart_patch,
            push,
            pop as pop_mock,
            local_rank,
            local_world_size,
        ):
            nvtx.switch_profile(
                iter_id=4,
                start=2,
                end=4,
                profile_ranks=list(range(8)),
                record_shape=False,
            )

        pop_mock.assert_called_once_with()
        cudart.cudaProfilerStop.assert_called_once_with()
        self.assertEqual(barrier_mock.call_count, 2)
        self.assertFalse(nvtx._PROFILER_ENABLED)

    def test_nvtx_cleanup_failure_does_not_skip_stop_or_barriers(self):
        (
            cudart,
            initialized,
            get_rank,
            barrier,
            cudart_patch,
            push,
            pop,
            local_rank,
            local_world_size,
        ) = self._distributed_mocks(rank=0, local_rank=0)
        pop_error = RuntimeError("range pop failed")
        with (
            initialized,
            get_rank,
            barrier as barrier_mock,
            cudart_patch,
            push,
            pop as pop_mock,
            local_rank,
            local_world_size,
        ):
            pop_mock.side_effect = pop_error
            with self.assertRaisesRegex(RuntimeError, "range pop failed"):
                nvtx.switch_profile(
                    iter_id=4,
                    start=2,
                    end=4,
                    profile_ranks=list(range(8)),
                    record_shape=False,
                )

        cudart.cudaProfilerStop.assert_called_once_with()
        self.assertEqual(barrier_mock.call_count, 2)
        self.assertFalse(nvtx._PROFILER_ENABLED)

    def test_start_precedes_barrier(self):
        calls = []
        (
            cudart,
            initialized,
            get_rank,
            barrier,
            cudart_patch,
            push,
            pop,
            local_rank,
            local_world_size,
        ) = self._distributed_mocks(rank=0, local_rank=0)
        cudart.cudaProfilerStart.side_effect = lambda: calls.append("start")
        with (
            initialized,
            get_rank,
            barrier as barrier_mock,
            cudart_patch,
            push,
            pop,
            local_rank,
            local_world_size,
        ):
            barrier_mock.side_effect = lambda: calls.append("barrier")
            nvtx.switch_profile(
                iter_id=2,
                start=2,
                end=4,
                profile_ranks=[1],
                record_shape=False,
            )

        self.assertEqual(calls, ["start", "barrier"])


if __name__ == "__main__":
    unittest.main()
