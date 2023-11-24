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

from ais_bench.evaluate.dataset.ceval_dataset import CevalDataset
from ais_bench.evaluate.dataset.mmlu_dataset import MmluDataset
from ais_bench.evaluate.dataset.gsm8k_dataset import Gsm8kDataset
from ais_bench.evaluate.log import logger

dataset_switch = {
    "ceval": CevalDataset,
    "mmlu": MmluDataset,
    "gsm8k": Gsm8kDataset
}


class DatasetFactory():
    def get(self, datasetname, dataset_path, shot):
        if dataset_switch.get(datasetname.strip()) is not None:
            return dataset_switch.get(datasetname.strip())(datasetname, dataset_path, shot)
        else:
            logger.error(f"Dataset {datasetname} is not supported."
                  f"Currently only {', '.join(list(dataset_switch.keys()))} are supported.")
            raise ValueError
