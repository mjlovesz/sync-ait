# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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

import os.path
import argparse

import pytest

from app_analyze.__main__ import get_cmd_instance


CUR_DIR = f"{os.path.dirname(__file__)}/"
SOURCE = os.path.join(CUR_DIR, "resources/opencv/")
REPORT_TYPE = "csv"
LOG_LEVEL = "INFO"
TOOLS = "cmake"
INVALID_ARG = "invalid_arg"


def transplt_argparse(argv):
    aa = get_cmd_instance()
    parser = argparse.ArgumentParser()
    aa.add_arguments(parser)
    return parser.parse_args(argv)


def call_transplt_cmd(argv):
    aa = get_cmd_instance()
    args = transplt_argparse(argv)
    return aa.handle(args)


def test_transplt_argparse_given_valid_when_short_then_pass():
    argv = ["-s", SOURCE, "-f", REPORT_TYPE, "--log-level", LOG_LEVEL, "--tools", TOOLS]
    transplt_argparse(argv)


def test_transplt_argparse_given_no_source_when_short_then_error():
    argv = ["-f", REPORT_TYPE, "--log-level", LOG_LEVEL, "--tools", TOOLS]
    with pytest.raises(SystemExit):
        transplt_argparse(argv)


def test_transplt_argparse_given_empty_source_when_short_then_error():
    argv = ["-s", "-f", REPORT_TYPE, "--log-level", LOG_LEVEL, "--tools", TOOLS]
    with pytest.raises(SystemExit):
        transplt_argparse(argv)


def test_transplt_argparse_given_invalid_report_type_when_short_then_error():
    argv = ["-s", SOURCE, "-f", INVALID_ARG, "--log-level", LOG_LEVEL, "--tools", TOOLS]
    with pytest.raises(SystemExit):
        transplt_argparse(argv)


def test_transplt_argparse_given_invalid_log_level_when_short_then_error():
    argv = ["-s", SOURCE, "-f", REPORT_TYPE, "--log-level", INVALID_ARG, "--tools", TOOLS]
    with pytest.raises(SystemExit):
        transplt_argparse(argv)


def test_transplt_argparse_given_invalid_tools_when_short_then_error():
    argv = ["-s", SOURCE, "-f", REPORT_TYPE, "--log-level", LOG_LEVEL, "--tools", INVALID_ARG]
    with pytest.raises(SystemExit):
        transplt_argparse(argv)


def test_transplt_argparse_given_valid_when_long_then_pass():
    argv = ["--source", SOURCE, "--report-type", REPORT_TYPE, "--log-level", LOG_LEVEL, "--tools", TOOLS]
    transplt_argparse(argv)


def test_transplt_argparse_given_no_source_when_long_then_error():
    argv = ["--report-type", REPORT_TYPE, "--log-level", LOG_LEVEL, "--tools", TOOLS]
    with pytest.raises(SystemExit):
        transplt_argparse(argv)


def test_transplt_argparse_given_empty_source_when_long_then_error():
    argv = ["--source", "--report-type", REPORT_TYPE, "--log-level", LOG_LEVEL, "--tools", TOOLS]
    with pytest.raises(SystemExit):
        transplt_argparse(argv)


def test_transplt_argparse_given_invalid_report_type_when_long_then_error():
    argv = ["--source", SOURCE, "--report-type", INVALID_ARG, "--log-level", LOG_LEVEL, "--tools", TOOLS]
    with pytest.raises(SystemExit):
        transplt_argparse(argv)


def test_app_analyze_given_opencv_csv_when_any_then_pass():
    argv = ["-s", SOURCE, "-f", REPORT_TYPE, "--log-level", LOG_LEVEL, "--tools", TOOLS]
    call_transplt_cmd(argv)

    output_xlsx = os.path.join(CUR_DIR, "resources/opencv/output.xlsx")
    assert os.path.exists(output_xlsx)
    os.remove(output_xlsx)


def test_app_analyze_given_invalid_source_when_any_then_error():
    argv = ["-s", INVALID_ARG, "-f", REPORT_TYPE, "--log-level", LOG_LEVEL, "--tools", TOOLS]
    with pytest.raises(argparse.ArgumentTypeError):
        call_transplt_cmd(argv)
