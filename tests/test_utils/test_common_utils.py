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
from unittest import TestCase

import torch

from magi_attention.utils import (
    flatten_nested_list,
    is_list_type_all,
    is_list_value_all,
    perm_idxs2unperm_idxs,
    transpose_matrix,
    wrap_to_list,
)


class TestCommonUtils(TestCase):
    def test_flatten_nested_list(self):
        # ---------    empty list     --------- #
        self.assertEqual(
            flatten_nested_list([]),
            [],
        )

        # ---------    empty tuple     --------- #
        self.assertEqual(
            flatten_nested_list(()),
            [],
        )

        # ---------    single int list     --------- #
        self.assertEqual(
            flatten_nested_list([1]),
            [1],
        )

        # ---------    single int tuple     --------- #
        self.assertEqual(
            flatten_nested_list((3,)),
            [3],
        )

        # ---------    1-dim int list     --------- #
        self.assertEqual(
            flatten_nested_list([1, 4, 5]),
            [1, 4, 5],
        )

        # ---------    1-dim int tuple     --------- #
        self.assertEqual(
            flatten_nested_list((2, 7, 9, 14, 16)),
            [2, 7, 9, 14, 16],
        )

        # ---------    2-dim int list     --------- #
        self.assertEqual(
            flatten_nested_list([[1, 2], [3, 4, 5], [6], [7, 8, 9, 10]]),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

        # ---------    2-dim int list embedded 1-dim int list/tuple     --------- #
        self.assertEqual(
            flatten_nested_list([[1, 2], 3, 4, (5, 6), [7, 8], (9,), [10]]),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

        # ---------    3-dim int list embedded 2-dim/1-dim int list/tuple     --------- #
        self.assertEqual(
            flatten_nested_list(
                [
                    [[1, 2], 3],
                    [4, (5, 6)],
                    [[7, 8], (9,)],
                    [[10]],
                ]
            ),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

    def test_perm_idxs2unperm_idxs(self):
        # ---------    empty perm idxs     --------- #

        self.assertEqual(
            perm_idxs2unperm_idxs([]),
            [],
        )

        # ---------    single-elem perm idxs     --------- #

        self.assertEqual(
            perm_idxs2unperm_idxs([0]),
            [0],
        )

        # ---------    4-elems perm idxs     --------- #

        self.assertEqual(
            perm_idxs2unperm_idxs([0, 1, 2, 3]),
            [0, 1, 2, 3],
        )

        self.assertEqual(
            perm_idxs2unperm_idxs([1, 2, 0, 3]),
            [2, 0, 1, 3],
        )

        self.assertEqual(
            perm_idxs2unperm_idxs([3, 1, 0, 2]),
            [2, 1, 3, 0],
        )

        self.assertEqual(
            perm_idxs2unperm_idxs([2, 1, 0, 3]),
            [2, 1, 0, 3],
        )

        # ---------    invalid perm idxs     --------- #

        with self.assertRaises(
            IndexError,
            msg="The input index permutation should start with 0",
        ):
            perm_idxs2unperm_idxs([1, 2, 3])

        with self.assertRaises(
            IndexError,
            msg="The input index permutation should be contiguous",
        ):
            perm_idxs2unperm_idxs([1, 0, 3])

    def test_wrap_to_list(self):
        # ---------    input an int     --------- #

        self.assertEqual(
            wrap_to_list(1),
            [1],
        )

        # ---------    input a float     --------- #

        self.assertEqual(
            wrap_to_list(2.5),
            [2.5],
        )

        # ---------    input a tensor     --------- #

        tensor = torch.randn(1, 2, 3)
        self.assertTrue(
            torch.equal(wrap_to_list(tensor)[0], tensor),
        )

        # ---------    input a list     --------- #

        self.assertEqual(
            wrap_to_list([1, 2.5, "3"]),
            [1, 2.5, "3"],
        )

        # ---------    input a tuple     --------- #

        self.assertEqual(
            wrap_to_list((2.7, "abc", 1)),
            [2.7, "abc", 1],
        )

        # ---------    input an int to broadcast     --------- #

        self.assertEqual(
            wrap_to_list(1, broadcast_to_length=4),
            [1, 1, 1, 1],
        )

        # ---------    input a tensor to broadcast     --------- #

        tensor = torch.randn(3, 2, 1)
        bc_len = 4
        tensor_list = wrap_to_list(tensor, broadcast_to_length=bc_len)
        self.assertEqual(len(tensor_list), bc_len)
        self.assertTrue(
            all(torch.equal(tensor, t) for t in tensor_list),
        )

    def test_is_list_value_all(self):
        # ----    empty list always False unless allow empty, then always True   ---- #

        self.assertFalse(is_list_value_all([], val=1))
        self.assertFalse(is_list_value_all([], just_same=True))
        self.assertFalse(is_list_value_all([], val=1, just_same=True))

        self.assertTrue(is_list_value_all([], val=1, allow_empty=True))
        self.assertTrue(is_list_value_all([], just_same=True, allow_empty=True))
        self.assertTrue(is_list_value_all([], val=1, just_same=True, allow_empty=True))

        # ----  error when val is given and just_same is True  ---- #

        with self.assertRaises(
            AssertionError,
            msg="The val should NOT be given, when just_same is True",
        ):
            is_list_value_all([1, 2, 3], val=4, just_same=True)

        # ---------    all same val list     --------- #

        self.assertTrue(is_list_value_all([2, 2, 2, 2], val=2))
        self.assertFalse(is_list_value_all([2, 2, 2, 2], val=3))

        # ---------    not-all same val list     --------- #

        self.assertFalse(is_list_value_all([2, 3, 4, 5], val=2))
        self.assertFalse(is_list_value_all([2, 3, 4, 5], val=1))

        # ---------    just-same     --------- #

        self.assertTrue(is_list_value_all([2, 2, 2, 2], just_same=True))
        self.assertFalse(is_list_value_all([2, 3, 4, 5], just_same=True))

    def test_is_list_type_all(self):
        # ----    empty list always False unless allow empty, then always True   ---- #

        self.assertFalse(is_list_type_all([], _type=1))
        self.assertFalse(is_list_type_all([], just_same=True))
        self.assertFalse(is_list_type_all([], _type=1, just_same=True))

        self.assertTrue(is_list_type_all([], _type=1, allow_empty=True))
        self.assertTrue(is_list_type_all([], just_same=True, allow_empty=True))
        self.assertTrue(is_list_type_all([], _type=1, just_same=True, allow_empty=True))

        # ----  error when _type is given and just_same is True  ---- #

        with self.assertRaises(
            AssertionError,
            msg="The _type should NOT be given, when just_same is True",
        ):
            is_list_type_all([1, 2, 3], _type=4, just_same=True)

        # ---------    a list of mixed types     --------- #

        self.assertTrue(is_list_type_all([1, "2", 3, "4"], _type=(int, str)))
        self.assertFalse(is_list_type_all([1, "2", 3.0, "4"], _type=(int, str)))

        # ---------    all same type list     --------- #

        self.assertTrue(is_list_type_all([2, 3, 4, 5], _type=int))
        self.assertFalse(is_list_type_all([2, 3, 4, 5], _type=float))
        self.assertTrue(is_list_type_all([2.0, 3.1, 4.2, 5.3], _type=float))
        self.assertTrue(
            is_list_type_all(
                [torch.randn(1), torch.randn(2), torch.randn(3, 4)],
                _type=torch.Tensor,
            )
        )

        # ---------    not-all same val list     --------- #

        self.assertFalse(
            is_list_type_all(
                [2.2, 1.4, 4.5, torch.randn(3, 4).sum()],
                _type=int,
            )
        )
        self.assertFalse(
            is_list_type_all(
                [2.2, 1.4, 4.5, torch.randn(3, 4).sum()],
                _type=float,
            )
        )

        # ---------    just-same     --------- #

        self.assertTrue(
            is_list_type_all(
                [
                    torch.randn(2, 4),
                    torch.randn(
                        3,
                    ),
                ],
                just_same=True,
            )
        )
        self.assertFalse(
            is_list_type_all(
                [2.2, 1.4, 4.5, torch.randn(3, 4).sum()],
                just_same=True,
            )
        )

    def test_transpose_matrix(self):
        class CustomObject:
            def __init__(self, value):
                self.value = value

            def __repr__(self) -> str:
                return f"CustomObject({self.value})"

            def __eq__(self, other) -> bool:
                if isinstance(other, CustomObject):
                    return self.value == other.value
                return False

        # create a 2D list with custom objects
        matrix = [
            [CustomObject(1), CustomObject(2), CustomObject(3)],
            [CustomObject(4), CustomObject(5), CustomObject(6)],
            [CustomObject(7), CustomObject(8), CustomObject(9)],
        ]
        ref_matrix_t = [
            [CustomObject(1), CustomObject(4), CustomObject(7)],
            [CustomObject(2), CustomObject(5), CustomObject(8)],
            [CustomObject(3), CustomObject(6), CustomObject(9)],
        ]

        # transpose the matrix
        matrix_t = transpose_matrix(matrix)

        self.assertEqual(matrix_t, ref_matrix_t)


if __name__ == "__main__":
    unittest.main()
