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

import re
import time
from contextlib import contextmanager

import torch

# -------------------       test utils     ------------------- #


class TimeManager:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        print(f"elapsed time: {self.elapsed_time:.3e}")


def info_str(center_content: str = "", side_str: str = "=", side_num: int = 15) -> str:
    return (
        "\n"
        + side_str * side_num
        + " "
        + center_content
        + " "
        + side_str * side_num
        + "\n"
    )


@contextmanager
def time_manager(name: str = ""):
    start_time = time.time()
    suffix = f" for {name}" if name != "" else ""
    try:
        print(info_str(f"Timing Begins{suffix}"))
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time

        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            info_str(
                f"Time Costed{suffix}: {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds"
            )
        )


def get_max_diff(
    a: torch.Tensor,
    b: torch.Tensor,
    ref: torch.Tensor | None = None,
    times: float = 1.0,
):
    """get the maximum difference between a and b if ref is None,
    otherwise get the maximum difference between (a-ref) and times * (b-ref)

    Args:
        a (tensor.Tensor): tensor a
        b (tensor.Tensor): tensor b
        ref (tensor.Tensor, optional): reference tensor. Defaults to None.
        times (float, optional): times of the (b-ref). Defaults to 1.0.

    Returns:
        diff (float): the maximum difference value
    """
    if ref is None:
        return (a - b).abs().max().item()
    return get_max_diff(a, ref) - times * get_max_diff(b, ref)


def get_mean_diff(a: torch.Tensor, b: torch.Tensor):
    return (a - b).abs().mean().item()


def is_allclose(a: torch.Tensor, b: torch.Tensor):
    if torch.allclose(a, b):
        print("✅ All close!!")
    else:
        print(
            f"❌ Not all close! The maximum difference is {torch.max(torch.abs(a - b))}"
        )


def compare_speed(s1: str, t1: float, s2: str, t2: float):
    if t1 > t2:
        print(f"The {s2} is {t1 / t2:.2f} times faster than {s1}!!")
    elif t1 < t2:
        print(f"The {s1} is {t2 / t1:.2f} times faster than {s2}!!")
    else:
        print(f"The {s1} and {s2} are the same fast!!")


def get_timestamp():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())


def get_device_name():
    device_name = torch.cuda.get_device_name()
    # search for the registered name patterns
    register_names = {
        ("3090",): "3090",
        ("3090ti", "3090Ti"): "3090ti",
        ("4090ti", "4090Ti"): "4090ti",
        ("4090",): "4090",
        ("A6000",): "A6000",
        ("A100",): "A100",
        ("A800",): "A800",
        ("H100",): "H100",
        ("H800",): "H800",
        ("V100",): "V100",
        ("T4",): "T4",
        ("P4",): "P4",
        ("P40",): "P40",
    }
    for ks, v in register_names.items():
        for k in ks:
            if k in device_name:
                return v

    # not regitered, default use the last word in the device name
    device_name_match = re.search(r"(\w+)$", device_name)
    if device_name_match:
        device_name = device_name_match.group(1)

    return device_name
