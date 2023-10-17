from ais_bench.evaluate.dataset.ceval_dataset import CevalDataset
from ais_bench.evaluate.dataset.mmlu_dataset import MmluDataset

dataset_switch = {
    "ceval": CevalDataset,
    "mmlu": MmluDataset
}

class DatasetFactory():
    def get(self, datasetname, dataset_path, shot):
        if dataset_switch.get(datasetname.strip()) is not None:
            return dataset_switch.get(datasetname.strip())(datasetname, dataset_path, shot)
        else:
            print(f"Dataset {datasetname} is not supported."
                  f"Currently only {', '.join(list(dataset_switch.keys()))} are supported.")
            raise ValueError
