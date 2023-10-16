from ais_bench.evaluate.dataset.dataset_factory import DatasetFactory
from ais_bench.evaluate.recorder import Recorder



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
        self.dataset = DatasetFactory().get_dataset(dataset_name, dataset_path, shot)

    def evaluate(self, measurement = None):
        recorder = Recorder()
        for index, entry_dict in self.dataset:
            answer = self.generate(entry_dict.get("prompt"))
            entry_dict["answer"] = answer
            if self.rank == 0:
                recorder.record(index, entry_dict)

        recorder.statistics(self.dataset.compute, measurement)
        recorder.report(self.dataset.report)
        return recorder
