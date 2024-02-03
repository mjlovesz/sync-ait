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

import pytest

from llm.compare import torchair_utils


FAKE_PBTXT_FILE_NAME = "test_torchair_utils_fake_pbtxt_file.txt"


@pytest.fixture(scope='module', autouse=True)
def fake_pbtxt_file():
    contents = """op: {
      name: "test"
      desc: {
        name: "test"
        attr: {
          name: "tt1"
        }
        attr: {
          name: "tt2"
        }
      }
    }"""

    with open(FAKE_PBTXT_FILE_NAME, "w") as ff:
        ff.write(contents)

    yield

    if os.path.exists(FAKE_PBTXT_FILE_NAME):
        os.emove(FAKE_PBTXT_FILE_NAME)


def test_parse_pbtxt_to_dict_given_path_when_valid_then_pass():
    result = torchair_utils.parse_pbtxt_to_dict(FAKE_PBTXT_FILE_NAME)
    assert isinstance(rr, dict)
    expected_result = [
        {'op:': {'name': 'test', 'desc:': {'name': 'test', 'attr:': {'name': 'tt1'}, 'attr:#1': {'name': 'tt2'}}}}
    ]
    assert result == expected_result


