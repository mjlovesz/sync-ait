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

import click.testing
import pytest

from app_analyze.__main__ import cli


@pytest.fixture
def runner():
    yield click.testing.CliRunner()


CUR_DIR = f'{os.path.dirname(__file__)}/'
SOURCE = os.path.join(CUR_DIR, 'resources/opencv/')
REPORT_TYPE = 'csv'
LOG_LEVEL = 'INFO'
TOOLS = 'cmake'


def test_app_analyze_opencv_csv(runner):
    runner.invoke(cli, ['-s', SOURCE,
                        '-f', REPORT_TYPE,
                        '--log-level', LOG_LEVEL,
                        '--tools', TOOLS])
    output_xlsx = os.path.join(CUR_DIR, 'resources/opencv/output.xlsx')
    assert os.path.exists(output_xlsx)
    os.remove(output_xlsx)
