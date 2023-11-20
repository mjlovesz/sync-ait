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

import pandas as pd

from ais_bench.evaluate.log import logger


class Recorder():
    def __init__(self, name="default") -> None:
        self.name = name
        self.records = None
        self.metrics = None
        self.children = dict()

    def record(self, index : list, entry_dict : dict):
        '''
        all the index should be of the same length
        '''
        if index == []:
            if self.records is None:
                self.records = pd.DataFrame(columns=entry_dict.keys())
            df_dict = pd.DataFrame([entry_dict])
            self.records = pd.concat([self.records, df_dict], ignore_index=True)
            return

        current_index = index[0]
        if current_index not in self.children:
            self.children[current_index] = Recorder(current_index)
        self.children.get(current_index).record(index[1:], entry_dict)

    def read(self, index):
        if index == []:
            return self.records
        current_index = index[0]
        return self.children.get(current_index).read(index[1:])

    def statistics(self, func_compute=None, measurement=None):
        if self.metrics is not None:
            return self.metrics
        if func_compute is None:
            logger.error("Record.statistics failed: function to compute metrics missing")
            raise Exception

        if not self.children:
            data = self.records
        else:
            data = []
            for child in self.children.values():
                data.append(child.statistics(func_compute, measurement))

        if measurement is None:
            # use default measurement
            self.metrics = func_compute(data)
        else:
            self.metrics = func_compute(data, measurement)
        return self.metrics

    def report(self):
        logger.info(self.metrics)