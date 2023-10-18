import os
import json
import pandas as pd
from ais_bench.evaluate.dataset.base_dataset import BaseDataset
from ais_bench.evaluate.measurement.measurement_factory import MeasurementFactory
from ais_bench.evaluate.log import logger
from ais_bench.infer.path_security_check import ms_open, MAX_SIZE_LIMITE_NORMAL_FILE

SUBCATEGORY_INDEX = 0
CHINESE_INDEX = 1
CATEGORY_INDEX = 2
VAL_INDEX = 3
PROMPT_INDEX = 4

class CevalDataset(BaseDataset):
    def _download(self):
        logger.error("please download the dataset and subject_mapping from "
                     "huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip "
                     "and github.com/SJTU-LIT/ceval/blob/main/subject-mapping.json.")
        raise ValueError

    def _check(self, dataset_path):
        pass

    def _gen_prompt(self, prompt_df, category_name):
        question_template = "问： {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答： {answer}\n"

        prompt = f"以下展示了在{category_name}领域的选择题及其正确答案\n\n"
        for _, row in prompt_df.iterrows():
            prompt += question_template.format(question=row["question"], A=row["A"], B=row["B"],
                                               C=row["C"], D=row["D"], answer=row["answer"])
        prompt += "请回答以下选择题\n"
        return prompt

    def load(self, dataset_path):
        '''
        dataset_path-
        use val dataset as validation use dev dataset as prompt
        '''
        if dataset_path is None:
            dataset_path = self._download()
        self._check(dataset_path)
        subject_mapping_path = os.path.join(dataset_path, "subject_mapping.json")
        with ms_open(subject_mapping_path, max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as file:
            self.subject_mapping = json.load(file)

        for subject in self.subject_mapping:
            val_path = os.path.join(dataset_path, "val", subject+"_val.csv")
            prompt_path = os.path.join(dataset_path, "dev", subject+"_dev.csv")
            with ms_open(val_path, max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as file:
                val_df = pd.read_csv(file, header=0)
            self.subject_mapping[subject].append(val_df)
            with ms_open(prompt_path, max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as file:
                prompt_df = pd.read_csv(file, header=0)[:self.shot+1]
            prompt = self._gen_prompt(prompt_df, self.subject_mapping[subject][CHINESE_INDEX])
            self.subject_mapping[subject].append(prompt)
        self.subjects = list(self.subject_mapping.keys())

    def __len__(self):
        count = 0
        for value in self.subject_mapping.values():
            count += len(value[VAL_INDEX])
        return count

    def __iter__(self):
        self.current_key = 0
        self.current_index = 0
        return self


    def __next__(self):
        if self.current_key >= len(self.subjects):
            raise StopIteration

        key = self.subjects[self.current_key]
        subcategory = self.subject_mapping[key][SUBCATEGORY_INDEX]
        category = self.subject_mapping[key][CATEGORY_INDEX]
        val_df  = self.subject_mapping[key][VAL_INDEX]
        prompt = self.subject_mapping[key][PROMPT_INDEX]

        if self.current_index >= len(val_df):
            self.current_key += 1
            self.current_index = 0
            return self.__next__()

        val_row = val_df.loc[self.current_index]

        prompt += f"问： {val_row['question']}\nA. {val_row['A']}\nB. {val_row['B']}\nC. {val_row['C']}\nD. {val_row['D']}\n答：\n"
        result = {"id": self.current_index, "category": category, "sub_category": subcategory,
                  "prompt": prompt, "ground_truth": val_row["answer"]}
        index = [category, subcategory]
        self.current_index += 1
        return index, result


    def compute(self, data, measurement = "accuracy") -> dict:
        '''
        input: data in the form of pandas.DataFrame OR a list of metrics dictonary
        output: a dictionary containing accuracy, total number of entry, number of correct entry
        '''
        ground_truth_index = "ground_truth"
        answer_index = "answer"
        measurement_method = MeasurementFactory().get(measurement)()

        output = measurement_method(data, ground_truth_index, answer_index)
        return output


    def report(self, metrics):
        '''
        input: a metrics (dictionary)
        output: None
        '''
        logger.info(metrics)



