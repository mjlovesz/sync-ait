from abc import abstractmethod, ABCMeta

class BaseDataset(metaclass=ABCMeta):
    def __init__(self, dataset_name, dataset_path, shot) -> None:
        self.dataset_name = dataset_name
        self.shot = shot
        self.load(dataset_path)

    @abstractmethod
    def load(self, dataset_path):
        # to do : open file or files in dataset_path and save as pd.dataframe
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, recorder): # dataset -> accuracy
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @abstractmethod
    def report(self, result):
        raise NotImplementedError
