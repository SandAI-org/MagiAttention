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

import unittest
from typing import Any
from unittest import mock

import pytest
from packaging.version import InvalidVersion, Version

from magi_attention.utils import version


@pytest.mark.arch_independent
class TestVersionComparisons(unittest.TestCase):
    def test_release_normalization(self) -> None:
        for text in ("2.12.0a0+nv26.05", "2.12.0rc1", "1!2.12.post2", "2.12.dev1"):
            with self.subTest(text=text):
                parsed = version._parse(text, release_only=True)
                self.assertIsInstance(parsed, Version)
                self.assertEqual(parsed, Version("2.12"))
                self.assertEqual(version._parse(text, False), Version(text))
        with self.assertRaises(InvalidVersion):
            version._parse("invalid", True)

    def test_all_components_agree_on_release_and_full_ordering(self) -> None:
        current = "2.12.0a0+nv26.05"
        patches: dict[str, Any] = {
            "check_cutedsl_installation": lambda: (current, current)
        }
        for component in ("TORCH", "TRITON", "CUDA"):
            patches[f"_{component}_VERSION"] = Version(current)
            patches[f"_{component}_RELEASE"] = Version("2.12.0")
        with mock.patch.multiple(version, **patches):
            for component in ("torch", "triton", "cuda", "cutedsl", "quack"):
                # Release comparison normalizes trailing zeros; full ordering
                # still distinguishes the installed prerelease from stable.
                expected = {
                    "eq": True,
                    "ne": False,
                    "ge": True,
                    "gt": False,
                    "le": True,
                    "lt": False,
                }
                for op, result in expected.items():
                    predicate = getattr(version, f"is_{component}_version_{op}", None)
                    if predicate is None:  # Triton does not export ne.
                        continue
                    for target in ("2.12", "2.12.0", "2.12.0.0"):
                        with self.subTest(component=component, op=op, target=target):
                            self.assertEqual(predicate(target), result)
                self.assertTrue(
                    getattr(version, f"is_{component}_version_lt")("2.12", False)
                )
                self.assertFalse(
                    getattr(version, f"is_{component}_version_eq")("2.12", False)
                )
                self.assertTrue(
                    getattr(version, f"is_{component}_version_eq")(current, False)
                )

    def test_missing_cuda_retains_unavailable_semantics(self) -> None:
        with mock.patch.multiple(version, _CUDA_VERSION=None, _CUDA_RELEASE=None):
            self.assertFalse(version.has_cuda_version())
            for op in ("eq", "ge", "gt", "le", "lt"):
                self.assertFalse(getattr(version, f"is_cuda_version_{op}")("12"))
            self.assertFalse(version.is_cuda_version_ne("12"))


if __name__ == "__main__":
    unittest.main()
