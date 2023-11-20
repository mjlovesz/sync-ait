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

from tqdm import tqdm

from ais_bench.evaluate.dataset.dataset_factory import DatasetFactory
from ais_bench.evaluate.recorder import Recorder
from ais_bench.evaluate.log import logger


class Evaluator():
    def __init__(self, generate_func, dataset_name, dataset_path=None, shot=0, rank=0):
        self.set_generate_func(generate_func)
        self.set_rank(rank)
        self.set_dataset(dataset_name, dataset_path, shot)

    def set_generate_func(self, generate_func):
        if callable(generate_func):
            all_args = generate_func.__code__.co_argcount
            kwargs = len(generate_func.__defaults__) if generate_func.__defaults__ else 0
            if all_args - kwargs != 1:
                logger.error("set_generate_func failed: number of function's required argument not equals to 1.")
                raise Exception
        else:
            logger.error("set_generate_func failed: please pass a callable object.")
            raise Exception
        self.generate = generate_func

    def set_rank(self, rank):
        self.rank = rank

    def set_dataset(self, dataset_name, dataset_path=None, shot=0):
        if isinstance(shot, int) and shot >= 0 and shot <= 5:
            self.dataset = DatasetFactory().get(dataset_name, dataset_path, shot)
            logger.info(f"Load dataset {dataset_name} success.")
        else:
            logger.error("set_dataset failed: shot should be of type int and between 0 and 5.")
            raise Exception

    def evaluate(self, measurement=None):
        logger.info(f"Start to evaluate on {self.dataset.dataset_name} dataset "
                    f"with {'default' if measurement is None else measurement} metric.")
        recorder = Recorder()
        for index, entry_dict in tqdm(self.dataset):
            answer = self.generate(entry_dict.get("prompt"))
            entry_dict["answer"] = answer
            if self.rank == 0:
                recorder.record(index, entry_dict)

        if self.rank == 0:
            recorder.statistics(self.dataset.compute, measurement)
            recorder.report()