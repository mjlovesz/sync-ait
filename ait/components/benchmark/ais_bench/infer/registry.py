# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd. All rights reserved.
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


import logging
from typing import Any, Dict, Iterable, Iterator, Tuple
from ais_bench.infer.utils import logger


class Registry(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    """
    def register(self, obj: Any = None) -> Any:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.See docstring of this class for usage.
        """
        if callable(obj):
            return add(None, obj)

        def add(name: str, obj: Any) -> Any:
            self[name] = obj
            return obj

        return lambda x: add(obj, x)
        
    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def __setitem__(self, name: str, obj: Any) -> None:
        if not callable(obj):
            raise ValueError("Value of a Registry must be a callable!")

        if name is None:
            name = obj.__name__

        if name in self._obj_map:
            raise ValueError(
                f"An object named '{name}' was already registered in '{self._name}' registry!"
            )
        self._obj_map[name] = obj

    def __getitem__(self, name: str) -> Any:
        return self._obj_map[name]

    def __call__(self, obj: Any) -> Any:
        return self.register(obj)

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        from tabulate import tabulate

        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())


def import_all_modules_for_register(module_paths, base_model_name):
    import os
    import importlib

    modules = []
    for _, _, files in os.walk(module_paths):
        for filename in files:
            if not filename.endswith(".py") or filename == "__init__.py":
                continue
            model_name = base_model_name + "." + filename.rsplit(".", 1)[0]
            modules.append(model_name)

    errors = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            errors.append((module, e))
            logger.info(f"import {module} error: {e}")

    return errors