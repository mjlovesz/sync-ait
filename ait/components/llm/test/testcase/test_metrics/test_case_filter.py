# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
from io import StringIO
import shutil
from unittest import TestCase

from ait_llm import CaseFilter


class TestCaseFilter(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.orig_stderr = sys.stderr
        cls.output_dir = os.path.dirname(__file__)

        cls.ins = ["Some random tests"]
        cls.outs = ["Some outputs"]
        cls.refs = ["References"]

    def setUp(self):
        self.case_filter = CaseFilter()
        self.captured_output = StringIO()

        self.temp_dir = ""
        self.temp_dir2 = ""

    def test_add_metrics_should_raise_when_key_not_valid(self):
        with self.assertRaises(RuntimeError):
            self.case_filter.add_metrics(key=3)

    def test_add_metrics_should_raise_when_value_not_valid(self):
        with self.assertRaises(RuntimeError):
            self.case_filter.add_metrics(accuracy=30)

        with self.assertRaises(RuntimeError):
            self.case_filter.add_metrics(edit_distance=-10)

    def test_add_metrics_should_not_empty_when_params_valid(self):
        self.case_filter.add_metrics(accuracy=None, edit_distance=None)

        self.assertTrue(self.case_filter._metrics)
        self.assertTrue(self.case_filter._columns)

    def test_apply_should_raise_when_lists_empty(self):
        self.case_filter.add_metrics(edit_distance=None)

        with self.assertRaises(RuntimeError):
            self.case_filter.apply(ins=[], outs=[], refs=[], output_dir=self.output_dir)
        
    def test_apply_should_raise_when_lists_diff_size(self):
        self.case_filter.add_metrics(accuracy=None)

        with self.assertRaises(RuntimeError):
            self.case_filter.apply(ins=[], outs=[""], refs=["", ""], output_dir=self.output_dir)

    def test_apply_should_raise_when_lists_not_string(self):
        self.case_filter.add_metrics(accuracy=None)

        with self.assertRaises(RuntimeError):
            self.case_filter.apply(ins=[1], outs=[2], refs=[3], output_dir=self.output_dir)

        self.case_filter.add_metrics(edit_distance=None)

        with self.assertRaises(RuntimeError):
            self.case_filter.apply(ins=[1], outs=[2], refs=[3], output_dir=self.output_dir)

    def test_apply_should_raise_when_apply_before_add_metrics(self):
        with self.assertRaises(RuntimeError):
            self.case_filter.apply(self.ins, self.outs, self.refs, output_dir=self.output_dir)

    def test_apply_should_raise_when_dir_not_valid(self):
        self.case_filter.add_metrics(accuracy=None)

        with self.assertRaises(TypeError):
            self.case_filter.apply(self.ins, self.outs, self.refs, output_dir=1)

        with self.assertRaises(TypeError):
            self.case_filter.apply(self.ins, self.outs, self.refs, output_dir=0.5)

        with self.assertRaises(RuntimeError):
            self.case_filter.apply(self.ins, self.outs, self.refs, output_dir="")

    def test_apply_should_warn_when_usr_not_owner(self):
        pass

    def test_apply_should_warn_when_dir_soft_link(self):
        self.temp_dir = "temp_dir"

        try:
            os.symlink(self.output_dir, self.temp_dir)
        except OSError as e:
            self.fail("fail to construct a soft link for test")

        sys.stderr = self.captured_output
        self.case_filter.add_metrics(accuracy=None)

        # do
        self.case_filter.apply(self.ins, self.outs, self.refs, output_dir=self.temp_dir)

        # check
        warn_message = ("Your attempt to use soft links as directories has triggered a security alert "
                        "and is being monitored closely for potential security threats.")
        self.assertTrue(warn_message in self.captured_output.getvalue())

    def test_apply_should_warn_when_no_enough_space(self):
        pass

    def test_apply_should_raise_when_usr_cannot_cd(self):
        # if root, do nothing
        if os.getuid() != 0:
            self.temp_dir = os.path.join(self.output_dir, "temp_dir")
            try:
                os.mkdir(self.temp_dir, 0o006)
            except OSError as e:
                self.fail(f"fail to make directory {temp_dir} under {current_dir}.")

            with self.assertRaises(PermissionError):
                self.case_filter.apply(self.ins, self.outs, self.refs, output_dir=self.temp_dir)

    def test_apply_should_raise_when_usr_cannot_write(self):
        # if root, do nothing
        if os.getuid() != 0:
            self.temp_dir = os.path.join(self.output_dir, "temp_dir")
            try:
                os.mkdir(temp_dir, 0o005)
            except OSError as e:
                self.fail(f"fail to make directory {temp_dir} under {current_dir}.")

            with self.assertRaises(PermissionError):
                self.case_filter.apply(self.ins, self.outs, self.refs, output_dir=self.temp_dir)

    def test_apply_should_raise_when_cannot_traverse(self):
        pass

    def test_apply_should_raise_when_dir_over_255(self):
        self.temp_dir = "a" * 256
        self.temp_dir2 = "b" * 999

        self.case_filter.add_metrics(accuracy=None)

        with self.assertRaises(RuntimeError):
            self.case_filter.apply(self.ins, self.outs, self.refs, output_dir=self.temp_dir)

        with self.assertRaises(RuntimeError):
            self.case_filter.apply(self.ins, self.outs, self.refs, output_dir=self.temp_dir2)

    def test_apply_should_raise_when_path_over_4095(self):
        self.temp_dir = "a" * 255
        for _ in range(20):
            self.temp_dir = os.path.join(self.temp_dir, self.temp_dir)

        self.case_filter.add_metrics(accuracy=None)

        with self.assertRaises(RuntimeError):
            self.case_filter.apply(self.ins, self.outs, self.refs, output_dir=self.temp_dir)

    def test_apply_should_not_raise_when_valid(self):
        self.case_filter.add_metrics(accuracy=None)
        self.temp_dir = os.path.join(self.output_dir, "temp_dir")

        self.case_filter.apply(self.ins, self.outs, self.refs, output_dir=self.temp_dir)
        
        self.assertTrue(os.path.isdir(self.temp_dir))
        self.assertEqual((os.stat(self.temp_dir).st_mode & 0o777), 0o750)

        all_files = os.listdir(self.temp_dir)
        self.assertTrue(any(file.endswith(".csv") for file in all_files)) # indeed create a csv file
        self.assertTrue(
            all(
                (os.stat(os.path.join(self.temp_dir, file)).st_mode & 0o777 == 0o640) for file in all_files if file.endswith(".csv")
            )
        ) # with permission 640
        
    def tearDown(self):
        sys.stderr = self.orig_stderr

        if os.path.exists(self.temp_dir):
            if os.path.islink(self.temp_dir):
                os.unlink(self.temp_dir)
            else:
                shutil.rmtree(self.temp_dir)
        
        if os.path.exists(self.temp_dir2):
            if os.path.islink(self.temp_dir2):
                os.unlink(self.temp_dir2)
            else:
                shutil.rmtree(self.temp_dir2)
    
    @classmethod
    def tearDownClass(cls):
        for file in os.listdir(os.path.dirname(__file__)):
            if file.endswith(".csv"):
                os.remove(file)
