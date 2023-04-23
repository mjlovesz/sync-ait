import unittest
import pytest
import sys
import os
from unittest import mock

sys.path.insert(0, os.path.abspath("../../../")) ##保证amit入口

from debug.compare.main import main, argsAdapter, _accuracy_compare_parser
from debug.compare.common import utils

@pytest.fixture(scope="module", autouse=True)
def fake_onnx_om():
    onnx_path = "/home/test/test.onnx"
    om_path = "/home/test/test.om"
    input_path = "/home/input/"

    return

def test_main_args_invalid_path(fake_onnx_om):
    with pytest.raises(SystemExit) as error:
        args = ["aaa.py", "-m", "/home/test/test.onnx", "-om", "/home/test/test.om"]
        with mock.patch('sys.argv', args):
            main()
    assert error.value.args[0] == utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR

def test_main_accuracy_compare_parser(fake_onnx_om):
    onnx_path = "/home/test/test.onnx"
    om_path = "/home/test/test.om"
    input_path = "/home/input/"

    args = ["aaa.py", "-m", onnx_path, "-om", om_path, "-i", input_path, "-c", "-d", "-s", "-o", "--output-size", "--output-nodes", "--advisor"]
    import argparse
    parser = argparse.ArgumentParser()
    _accuracy_compare_parser(parser)


def test_main_args_adapter(fake_onnx_om):
    onnx_path = "/home/test/test.onnx"
    om_path = "/home/test/test.om"
    input_path = "/home/input/"

    args = ["aaa.py", "-m", onnx_path, "-om", om_path, "-i", input_path]
    import argparse
    parser = argparse.ArgumentParser()
    _accuracy_compare_parser(parser)
    args = parser.parse_args(args[1:])
    args.model_path = os.path.realpath(args.model_path)

    my_args = argsAdapter(args)
    assert my_args.model_path == onnx_path
    assert my_args.offline_model_path == om_path
    assert my_args.input_path== input_path