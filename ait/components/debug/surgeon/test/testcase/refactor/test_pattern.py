# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd. All rights reserved.
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

import unittest

from auto_optimizer.pattern.pattern import MatchPattern
from auto_optimizer.pattern.pattern import Pattern


class TestPattern(unittest.TestCase):

    def test_add_node_func_0(self):
        pattern = Pattern()
        pattern.add_node('Conv_0', ['Conv'], None)
        pattern.add_node('Conv_1', ['Conv'], None)

        self.assertEqual(len(pattern.node_dict), 2)
        self.assertNotEqual(pattern.node_dict.get('Conv_0'), None)
        self.assertNotEqual(pattern.node_dict.get('Conv_1'), None)

    def test_add_node_func_1(self):
        pattern = Pattern()

        try:
            pattern.add_node('Conv', ['Conv'], None)
        except RuntimeError as e:
            pass
        try:
            pattern.add_node('Conv', ['Conv'], None)
        except RuntimeError as e:
            pass

        self.assertEqual(len(pattern.node_dict), 1)
        self.assertNotEqual(pattern.node_dict.get('Conv'), None)

    def test_node_can_match_more_func(self):
        pattern = Pattern() \
            .add_node('Conv', ['Conv'], None) \
            .add_node('Relu', ['Relu'], None) \
            .add_edge('Conv', 'Relu') \
            .set_node_loop('Conv', MatchPattern.MATCH_ONCE_OR_MORE) \
            .set_node_loop('Relu', MatchPattern.MATCH_ZERO_OR_MORE)

        self.assertTrue(pattern.node_dict['Conv'].can_match_more_time())
        self.assertTrue(pattern.node_dict['Relu'].can_match_more_time())

    def test_node_can_match_zero_func(self):
        pattern = Pattern() \
            .add_node('Conv', ['Conv'], None) \
            .add_node('Relu', ['Relu'], None) \
            .add_edge('Conv', 'Relu') \
            .set_node_loop('Conv', MatchPattern.MATCH_ONCE_OR_MORE) \
            .set_node_loop('Relu', MatchPattern.MATCH_ZERO_OR_MORE)

        self.assertFalse(pattern.node_dict['Conv'].can_match_zero_time())
        self.assertTrue(pattern.node_dict['Relu'].can_match_zero_time())

    def test_cann_match_more_func(self):
        pattern = Pattern() \
            .add_node('Conv', ['Conv'], None) \
            .add_node('Relu', ['Relu'], None) \
            .add_edge('Conv', 'Relu') \
            .set_loop(MatchPattern.MATCH_ONCE_OR_MORE)

        self.assertTrue(pattern.can_match_more())


if __name__ == "__main__":
    unittest.main()
