from ais_bench.evaluate.dataset.dataset_factory import DatasetFactory
from ais_bench.evaluate.recorder import Recorder
from ais_bench.evaluate.log import logger
from tqdm import tqdm




class Evaluator():
    def __init__(self, generate_func, dataset_name, dataset_path = None, shot = 0, rank = 0):
        self.set_generate_func(generate_func)
        self.set_rank(rank)
        self.set_dataset(dataset_name, dataset_path, shot)

    def set_generate_func(self, generate_func):
        self.generate = generate_func

    def set_rank(self, rank):
        self.rank = rank

    def set_dataset(self, dataset_name, dataset_path, shot):
        self.dataset = DatasetFactory().get(dataset_name, dataset_path, shot)
        logger.info(f"Load dataset {dataset_name} success.")

    def evaluate(self, measurement = None):
        logger.info(f"Start to evaluate on {self.dataset.dataset_name} dataset\
                    with {'default' if measurement is None else measurement} metric.")
        recorder = Recorder(rank=self.rank)
        for index, entry_dict in tqdm(self.dataset):
            answer = self.generate(entry_dict.get("prompt"))
            entry_dict["answer"] = answer
            recorder.record(index, entry_dict)

        recorder.statistics(self.dataset.compute, measurement)
        recorder.report(self.dataset.report)
        return recorder
