from ais_bench.evaluate.dataset.ceval_dataset import CevalDataset

class DatasetFactory():
    def get_dataset(self, datasetname, dataset_path, shot):
        if datasetname == "ceval":
            return CevalDataset(datasetname, dataset_path, shot)
        else:
            print(f"dataset {datasetname} is not supported")
            raise ValueError
