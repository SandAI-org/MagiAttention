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

import threading
from typing import Any


class SingletonMeta(type):
    """
    This is a generic multi-thread-safe singleton metaclass.

    Example:
        ```
        class A(metaclass=SingletonMeta):
            # Your class definition here
            pass

        # Create instances of class A
        instance1 = A()
        instance2 = A()

        # Check if both instances refer to the same object
        print(instance1 is instance2)  # Output: True
        ```

    """

    _instances: dict[type, Any] = {}

    __lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        # Check whether cls is in dict. If not, get the lock and create it
        if cls not in cls._instances:
            with cls.__lock:
                cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]
