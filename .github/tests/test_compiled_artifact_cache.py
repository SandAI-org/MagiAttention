#!/usr/bin/env python3

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

"""Fast synthetic tests for the MagiAttention compiled artifact cache."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compiled_artifact_cache.py"
SPEC = importlib.util.spec_from_file_location("compiled_artifact_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class CompiledArtifactCacheTest(unittest.TestCase):
    def test_publish_verify_and_restore(self) -> None:
        inputs = {"schema_version": cache.SCHEMA, "test": "inputs"}
        identity = cache.fingerprint(inputs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "magi_attention.whl"
            cache_root = root / "cache"
            destination = root / "destination"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("magi_attention/_C.so", b"native")
                archive.writestr("magi_attention/lib/ffa/aot.so", b"aot")
                archive.writestr("magi_attention/frontend.py", b"python")
                archive.writestr("other_package/bad.so", b"other")

            with mock.patch.object(
                cache,
                "artifact_dir",
                return_value=(cache_root / identity, inputs, identity),
            ):
                cache.publish(root, wheel, cache_root)
                self.assertTrue(cache.restore(root, destination, cache_root))

            self.assertEqual(
                (destination / "magi_attention/_C.so").read_bytes(), b"native"
            )
            self.assertEqual(
                (destination / "magi_attention/lib/ffa/aot.so").read_bytes(), b"aot"
            )
            self.assertFalse((destination / "magi_attention/frontend.py").exists())
            self.assertFalse((destination / "other_package/bad.so").exists())

    def test_corrupt_artifact_is_rejected(self) -> None:
        inputs = {"schema_version": cache.SCHEMA, "test": "inputs"}
        identity = cache.fingerprint(inputs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "magi_attention.whl"
            cache_root = root / "cache"
            artifact_dir = cache_root / identity
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("magi_attention/_C.so", b"native")

            with mock.patch.object(
                cache, "artifact_dir", return_value=(artifact_dir, inputs, identity)
            ):
                cache.publish(root, wheel, cache_root)
                (artifact_dir / "magi_attention/_C.so").write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "verification failed"):
                    cache.restore(root, root / "destination", cache_root)

    def test_binary_member_allowlist(self) -> None:
        self.assertTrue(cache.binary_member("magi_attention/_C.so"))
        self.assertTrue(cache.binary_member("magi_attention/lib/ffa/aot.so"))
        self.assertFalse(cache.binary_member("magi_attention/frontend.py"))
        self.assertFalse(cache.binary_member("other/_C.so"))
        self.assertFalse(cache.binary_member("magi_attention/lib/aot.so"))


if __name__ == "__main__":
    unittest.main()
