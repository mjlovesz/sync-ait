import os
import json
import pandas as pd
from ais_bench.evaluate.dataset.base_dataset import BaseDataset
from ais_bench.evaluate.measurement.measurement_factory import MeasurementFactory
from ais_bench.infer.path_security_check import ms_open, MAX_SIZE_LIMITE_NORMAL_FILE


class MmluDataset(BaseDataset):
    def load(self, dataset_path):
        if dataset_path is None:
            dataset_path = self._download()
        self.subject_mapping = dict()
        for root, _, files in os.walk(os.path.join(dataset_path, "val")):
            for file in files:
                subject_name = file.strip().strip("_val.csv")
                val_path = os.path.join(root, file)
                with ms_open(val_path, max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as file:
                    val_df = pd.read_csv(file, header=None)
                self.subject_mapping[subject_name] = [val_df]

        for root, _, files in os.walk(os.path.join(dataset_path, "test")):
            for file in files:
                subject_name = file.strip().strip("_test.csv")
                test_path = os.path.join(root, file)
                with ms_open(test_path, max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as file:
                    test_df = pd.read_csv(file, header=None)
                self.subject_mapping[subject_name].append(test_df)
        self.subjects = list(self.subject_mapping.keys())

    def __iter__(self):
        self.current_key = 0
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_key >= len(self.subjects):
            raise StopIteration

        key = self.subjects[self.current_key]
        prompt_df = self.subject_mapping[key][0]
        test_df = self.subject_mapping[key][1]


        if self.current_index >= len(test_df):
            self.current_key += 1
            self.current_index = 0
            return self.__next__()

        prompt = self._gen_prompt(prompt_df, category, val_df.loc[self.current_index])
        result = {"id": self.current_index, "category": category, "sub_category": subcategory,
                  "prompt": prompt, "ground_truth": val_df.loc[self.current_index, "answer"]}
        index = [category, subcategory]
        self.current_index += 1
        return index, result
