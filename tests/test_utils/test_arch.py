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

"""Architecture aliases, exact matching, and device-aware cache contracts."""

import unittest
from types import SimpleNamespace
from unittest import TestCase, mock

import pytest
import torch

from magi_attention.utils import arch

TEST_SMS = (75, 80, 86, 87, 88, 89, 90, 91, 100, 103, 107, 109, 110, 120, 121, 129, 130)


@pytest.mark.arch_independent
class TestArchitecturePredicates(TestCase):
    def setUp(self):
        arch.clear_arch_cache()
        self.addCleanup(arch.clear_arch_cache)

    def test_aliases_exclude_unqualified_and_future_capabilities(self):
        expected = {
            "ampere": {80, 86, 87, 88},
            "ada": {89},
            "hopper": {90},
            "blackwell": {100, 103},
            "rubin": {107},
        }
        for capability in TEST_SMS:
            with (
                self.subTest(capability=capability),
                mock.patch.object(arch, "get_dev_cap_num", return_value=capability),
            ):
                for name, sms in expected.items():
                    self.assertEqual(getattr(arch, f"is_{name}")(), capability in sms)

    def test_exact_matching_accepts_multiple_sms_and_unknown_numeric_values(self):
        with mock.patch.object(arch, "get_dev_cap_num", return_value=103):
            for sms in [(103,), ("103",), ("sm103",), ("SM103",), ("100", "103")]:
                with self.subTest(sms=sms):
                    self.assertTrue(arch.is_sm(*sms))
            self.assertFalse(arch.is_sm(100, 120))
        with mock.patch.object(arch, "get_dev_cap_num", return_value=137):
            self.assertTrue(arch.is_sm("137"))
            self.assertFalse(arch.is_blackwell())
        with self.assertRaises(ValueError):
            arch.is_sm()
        for target in ("sm100a", "sm100f", "10.3", "10x"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                arch.is_sm(target)

    def test_alias_membership_is_not_separately_cached(self):
        with mock.patch.object(arch, "get_dev_cap_num", return_value=130):
            self.assertFalse(arch.is_blackwell())
            with mock.patch.dict(arch.ARCH_SMS, blackwell=frozenset({130})):
                self.assertTrue(arch.is_blackwell())
            self.assertFalse(arch.is_blackwell())

    def test_predicates_forward_device_and_flush(self):
        predicates = [
            arch.is_ampere,
            arch.is_ada,
            arch.is_hopper,
            arch.is_blackwell,
            arch.is_rubin,
        ]
        for predicate in predicates:
            with (
                self.subTest(predicate=predicate.__name__),
                mock.patch.object(arch, "get_dev_cap_num", return_value=120) as query,
            ):
                predicate(device="cuda:2", flush=True)
                query.assert_called_once_with("cuda:2", flush=True)
        with mock.patch.object(arch, "get_dev_cap_num", return_value=103) as query:
            self.assertTrue(arch.is_sm("103", device=2, flush=True))
            query.assert_called_once_with(2, flush=True)

    def test_capability_query_is_shared_across_predicates_and_device_spellings(self):
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ) as query:
            self.assertTrue(arch.is_blackwell(2))
            self.assertTrue(arch.is_sm("103", device="cuda:2"))
            self.assertTrue(arch.is_sm("103", device=torch.device("cuda:2")))
            self.assertEqual(arch.get_dev_cap_str(device=2), "sm103")
            query.assert_called_once_with(2)
            self.assertEqual(arch.get_dev_cap_num(1), 103)
            self.assertEqual(query.call_count, 2)

    def test_current_device_is_resolved_before_cache_lookup(self):
        with (
            mock.patch.object(torch.cuda, "current_device", return_value=1) as current,
            mock.patch.object(
                torch.cuda,
                "get_device_capability",
                side_effect=lambda index: {1: (9, 0), 2: (10, 3)}[index],
            ) as query,
        ):
            self.assertEqual(arch.get_dev_cap_num(None), 90)
            self.assertTrue(arch.is_hopper("cuda"))
            current.return_value = 2
            self.assertEqual(arch.get_dev_cap_num(None), 103)
            self.assertTrue(arch.is_blackwell(torch.device("cuda")))
            current.return_value = 1
            self.assertEqual(arch.get_dev_cap_num(None), 90)
            self.assertEqual(query.call_args_list, [mock.call(1), mock.call(2)])

    def test_legacy_default_device_and_string_argument_order(self):
        with (
            mock.patch.object(torch.cuda, "current_device") as current,
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 3)
            ) as query,
        ):
            self.assertEqual(arch.get_dev_cap_num(), 103)
            self.assertEqual(arch.get_dev_cap_str(True, 0), "SM103")
            self.assertTrue(arch.is_blackwell())
            current.assert_not_called()
            query.assert_called_once_with(0)

    def test_sm_count_is_cached_per_device(self):
        with mock.patch.object(
            torch.cuda,
            "get_device_properties",
            side_effect=lambda index: SimpleNamespace(
                multi_processor_count={0: 132, 2: 148}[index]
            ),
        ) as query:
            self.assertEqual(arch.get_num_sms(), 132)
            self.assertEqual(arch.get_num_sms(2), 148)
            self.assertEqual(arch.get_num_sms("cuda:2"), 148)
            self.assertEqual(query.call_args_list, [mock.call(0), mock.call(2)])

    def test_flush_refreshes_queries_and_restores_values_after_a_mock(self):
        with (
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 3)
            ) as capability,
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=148),
            ) as properties,
        ):
            self.assertTrue(arch.is_blackwell())
            self.assertEqual(arch.get_num_sms(), 148)
            with (
                mock.patch.object(
                    torch.cuda, "get_device_capability", return_value=(9, 0)
                ),
                mock.patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=SimpleNamespace(multi_processor_count=132),
                ),
            ):
                self.assertTrue(
                    arch.is_blackwell()
                )  # Cached until explicitly refreshed.
                self.assertTrue(arch.is_hopper(flush=True))
                self.assertEqual(
                    arch.get_num_sms(), 132
                )  # Flush also cleared the SM-count cache.
            self.assertTrue(arch.is_hopper())
            self.assertEqual(arch.get_num_sms(flush=True), 148)
            self.assertTrue(arch.is_blackwell())
            self.assertEqual(capability.call_count, 2)
            self.assertEqual(properties.call_count, 2)

    def test_each_query_can_force_a_fresh_runtime_lookup(self):
        queries = [
            arch.get_dev_cap_num,
            arch.get_dev_cap_str,
            arch.is_ampere,
            arch.is_ada,
            arch.is_hopper,
            arch.is_blackwell,
            arch.is_rubin,
            lambda **kwargs: arch.is_sm(103, **kwargs),
        ]
        for predicate in queries:
            arch.clear_arch_cache()
            with (
                self.subTest(predicate=predicate),
                mock.patch.object(
                    torch.cuda, "get_device_capability", return_value=(10, 3)
                ) as query,
            ):
                predicate()
                predicate()
                predicate(flush=True)
                predicate()
                self.assertEqual(query.call_count, 2)

    def test_failed_runtime_query_does_not_poison_the_cache(self):
        with mock.patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=[RuntimeError("query failed"), (10, 3)],
        ) as query:
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                arch.get_dev_cap_num(0)
            self.assertEqual(arch.get_dev_cap_num(0), 103)
            self.assertEqual(arch.get_dev_cap_num(0), 103)
            self.assertEqual(query.call_count, 2)

    def test_non_cuda_devices_are_rejected(self):
        for device in ("cpu", torch.device("cpu")):
            with (
                self.subTest(device=device),
                self.assertRaisesRegex(ValueError, "CUDA device"),
            ):
                arch.get_dev_cap_num(device)


if __name__ == "__main__":
    unittest.main()
