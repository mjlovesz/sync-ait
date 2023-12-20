# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentTypeError
import os
import stat
import pytest

from llm.common.utils import (
    check_positive_integer,
    safe_string,
    check_number_list,
    check_ids_string,
    check_exec_script_file,
    check_input_args,
    check_exec_cmd,
    check_output_path_legality,
    check_input_path_legality,
    check_data_file_size,
    str2bool,
)


@pytest.fixture(scope='module')
def temp_large_file(tmp_path_factory):
    file_path = tmp_path_factory.mktemp("data") / "data_file.txt"
   # 文件权限为 640 (-rw-r-----)
    file_permissions = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP

    # 创建文件并指定权限

    # 使用文件描述符创建文件对象
    with os.fdopen(os.open(file_path, os.O_CREAT | os.O_WRONLY, file_permissions), 'wb') as f:
        f.write(b'hello')
    
    return str(file_path)


# Test cases for check_positive_integer function
@pytest.mark.parametrize("value, expected", [(1, 1), (0, 0), (2, 2)])
def test_check_positive_integer_valid(value, expected):
    assert check_positive_integer(value) == expected


@pytest.mark.parametrize("value", [-1, 3])
def test_check_positive_integer_invalid(value):
    with pytest.raises(ArgumentTypeError):
        check_positive_integer(value)


# Test cases for safe_string function
@pytest.mark.parametrize("value", ["ValidString123", "", None])
def test_safe_string_valid(value):
    assert safe_string(value) == value


@pytest.mark.parametrize("value", ["Invalid|String"])
def test_safe_string_invalid(value):
    with pytest.raises(ValueError):
        safe_string(value)


# Test cases for check_number_list function
@pytest.mark.parametrize("value, expected", [("1,2,3", "1,2,3"), ("", ""), (None, None)])
def test_check_number_list_valid(value, expected):
    assert check_number_list(value) == expected


@pytest.mark.parametrize("value", ["1,2,invalid", "string", "1,2,3,invalid"])
def test_check_number_list_invalid(value):
    with pytest.raises(ArgumentTypeError):
        check_number_list(value)


# Test cases for check_ids_string function
@pytest.mark.parametrize("value, expected", [("1_2,3_4", "1_2,3_4"), ("", ""), (None, None)])
def test_check_ids_string_valid(value, expected):
    assert check_ids_string(value) == expected


@pytest.mark.parametrize("value", ["invalid_ids", "-1,-1", "_1_0,1_0"])
def test_check_ids_string_invalid(value):
    with pytest.raises(ArgumentTypeError):
        check_ids_string(value)


# Test cases for check_exec_script_file function
def test_check_exec_script_file_existing_file():
    with pytest.raises(ArgumentTypeError):
        check_exec_script_file("non_existing_script.sh")


# Test cases for check_input_args function
@pytest.mark.parametrize("args", [["arg1", "|", "arg3"]])
def test_check_input_args(args):
    with pytest.raises(ArgumentTypeError):
        check_input_args(args)


# Test cases for check_exec_cmd function
@pytest.mark.parametrize("command", ["python3 aa.sh", "invalid command"])
def test_check_exec_cmd(command):
    with pytest.raises(ArgumentTypeError):
        check_exec_cmd(command)


# Test cases for check_output_path_legality function
def test_check_output_path_legality_existing_path():
    with pytest.raises(ArgumentTypeError):
        check_output_path_legality("invalid_&&_file|path")


# Test cases for check_input_path_legality function
def test_check_input_path_legality_existing_paths():
    with pytest.raises(ArgumentTypeError):
        check_input_path_legality("non_existing_input_path1,non_existing_input_path2")


# Test cases for check_data_file_size function
def test_check_data_file_size_existing_legal_file(temp_large_file):
    assert check_data_file_size(temp_large_file) == True


def test_check_data_file_size_non_existing_file():
    non_existing_file = "non_existing_file.txt"
    assert not os.path.exists(non_existing_file)
    with pytest.raises(Exception):
        check_data_file_size(non_existing_file)


@pytest.mark.parametrize("value, expected", [
    ("True", True), ("true", True), ("T", True), ("t", True), ("1", True),
    ("False", False), ("false", False), ("F", False), ("f", False), ("0", False),
])
def test_str2bool_valid(value, expected):
    assert str2bool(value) == expected


@pytest.mark.parametrize("value", ["invalid", "2", ""])
def test_str2bool_invalid(value):
    with pytest.raises(ArgumentTypeError):
        str2bool(value)