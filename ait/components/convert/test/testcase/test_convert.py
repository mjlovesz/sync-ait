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
import os
import sys
import unittest
from unittest import mock
import argparse

import pytest
import torch
from torch import nn

from model_convert.aie.bean import ConvertConfig
from model_convert.aie.core.convert import Convert
from model_convert.cmd_utils import add_arguments, gen_convert_cmd, execute_cmd
from model_convert.__main__ import get_cmd_instance, BaseCommand, ModelConvertCommand, AieCommand


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, (3, 3))

    def forward(self, x):
        return self.conv(x)


class TestConvert(unittest.TestCase):
    def setUp(self) -> None:
        model = TestModel()
        dummy_input = torch.randn((1, 3, 32, 32))
        torch.onnx.export(model, dummy_input, "test.onnx")
        assert os.path.exists("test.onnx")

    def tearDown(self) -> None:
        if os.path.exists("test.onnx"):
            os.remove("test.onnx")

    def test_convert(self):
        config = ConvertConfig("test.onnx", "test.om", "Ascend310")
        convert = Convert(config)
        convert.execute_command = mock.Mock(return_value="Execute command success.")
        convert.convert_model()

    def test_get_cmd_instance_when_valid_case(self):
        convert_cmd = get_cmd_instance()
        assert isinstance(convert_cmd, BaseCommand)
        assert len(convert_cmd.children) == 3
        assert isinstance(convert_cmd.children[0], ModelConvertCommand)
        assert isinstance(convert_cmd.children[1], ModelConvertCommand)
        assert isinstance(convert_cmd.children[2], AieCommand)

    def test_add_arguments_when_backend_atc(self):
        parser = argparse.ArgumentParser()
        args = add_arguments(parser, backend="atc")
        assert isinstance(args, list)
        assert args[0].get("name") == "--mode"

    def test_add_arguments_when_backend_aoe(self):
        parser = argparse.ArgumentParser()
        args = add_arguments(parser, backend="aoe")
        assert isinstance(args, list)
        assert args[0].get("name") == "--model"

    def test_add_arguments_when_invalid_backend(self):
        parser = argparse.ArgumentParser()
        with pytest.raises(ValueError):
            add_arguments(parser, backend="invalid_backend")

    def test_gen_convert_cmd_when_backend_atc(self):
        parser = argparse.ArgumentParser()
        conf_args = add_arguments(parser, backend="atc")
        if len(sys.argv) > 1:
            sys.argv = sys.argv[:1]
        else:
            sys.argv.append("test")
        sys.argv.extend(
            ["--model", 'test.onnx', '--framework', '5', '--soc_version', 'Ascend310P3', '--output', 'test']
        )
        args = parser.parse_args()
        cmds = gen_convert_cmd(conf_args, args, backend="atc")
        real_model_path = os.path.abspath("test.onnx")
        real_output = os.path.abspath("test")
        assert cmds == [
            "atc",
            '--model=' + real_model_path,
            '--framework=5',
            '--output=' + real_output,
            '--soc_version=Ascend310P3',
        ]

    def test_gen_convert_cmd_when_backend_aoe(self):
        parser = argparse.ArgumentParser()
        conf_args = add_arguments(parser, backend="aoe")
        if len(sys.argv) > 1:
            sys.argv = sys.argv[:1]
        else:
            sys.argv.append("test")
        sys.argv.extend(["--model", 'test.onnx', '--job_type', '1', '--framework', '5', '--output', 'test'])
        args = parser.parse_args()
        cmds = gen_convert_cmd(conf_args, args, backend="aoe")
        real_model_path = os.path.abspath("test.onnx")
        real_output = os.path.abspath("test")
        assert cmds == ["aoe", '--model=' + real_model_path, '--framework=5', '--job_type=1', '--output=' + real_output]

    def test_gen_convert_cmd_when_backend_invalid_backend(self):
        parser = argparse.ArgumentParser()
        conf_args = add_arguments(parser, backend="aoe")
        if len(sys.argv) > 1:
            sys.argv = sys.argv[:1]
        else:
            sys.argv.append("test")
        sys.argv.extend(
            [
                "--model",
                'test.onnx',
                '--job_type',
                '1',
            ]
        )
        args = parser.parse_args()
        with pytest.raises(ValueError):
            gen_convert_cmd(conf_args, args, backend="invalid")

    def test_execute_cmd_when_valid_cmd(self):
        cmd = ['pwd']
        ret = execute_cmd(cmd)
        assert ret == 0

    def test_execute_cmd_when_invalid_cmd(self):
        cmd = ['cp', 'test_execute_cmd.txt', './']
        ret = execute_cmd(cmd)
        assert ret != 0
