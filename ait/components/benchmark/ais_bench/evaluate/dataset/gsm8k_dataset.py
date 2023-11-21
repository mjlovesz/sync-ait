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
import json
import re

from ais_bench.evaluate.dataset.base_dataset import BaseDataset
from ais_bench.evaluate.measurement.measurement_factory import MeasurementFactory
from ais_bench.evaluate.log import logger
from ais_bench.infer.path_security_check import ms_open, MAX_SIZE_LIMITE_NORMAL_FILE

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


class Gsm8kDataset(BaseDataset):
    def _extract_ground_truth(self, answer):
        match = ANS_RE.search(answer)
        if match:
            match_str = match.group(1).strip()
            match_str.replace(",", "")
            return match_str
        else:
            logger.error(f"Extracting groud truth from dataset failed. Raw answer is {answer}")
            raise ValueError

    def load(self):
        if self.dataset_path is None:
            self._download()
        self._check()

        train_path = os.path.join(self.dataset_path, "train.jsonl")
        test_path = os.path.join(self.dataset_path, "test.jsonl")
        self.validation = []
        prompt_list = []
        prompt_count = 0
        with ms_open(train_path, max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as file:
            for line in file:
                data = json.loads(line)
                if prompt_count < self.shot:
                    prompt_list.append(data)
                    prompt_count += 1
                else:
                    self.validation.append(data)

        with ms_open(test_path, max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as file:
            for line in file:
                data = json.loads(line)
                if prompt_count < self.shot:
                    prompt_list.append(data)
                    prompt_count += 1
                else:
                    self.validation.append(data)

        question_template = "Question: {question}\nAnswer: {answer}\n"
        self.prompt = f"The following are grade school math questions (with answer).\n\n"
        for prompt_dict in prompt_list:
            self.prompt += question_template.format(question=prompt_dict.get("question"),
                                               answer=self._extract_ground_truth(prompt_dict.get("answer")))
        self.prompt += "Please answer the following questions.\n"

    def __len__(self):
        return len(self.validation)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.validation):
            raise StopIteration

        prompt = self.prompt + f"Question: {self.validation[self.current_index].get('question')}\nAnswer:\n"
        result = {"id": self.current_index, "prompt": prompt,
                  "ground_truth": self._extract_ground_truth(self.validation[self.current_index].get("answer"))}
        index = []
        self.current_index += 1
        return index, result

    def compute(self, data, measurement="accuracy") -> dict:
        '''
        input: data in the form of pandas.DataFrame OR a list of metrics dictonary
        output: a dictionary containing accuracy, total number of entry, number of correct entry
        '''
        ground_truth_index = "ground_truth"
        answer_index = "answer"
        measurement_method = MeasurementFactory().get(measurement)()

        output = measurement_method(data, ground_truth_index, answer_index)
        return output