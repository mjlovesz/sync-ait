import os
import subprocess
import hashlib
import json
from abc import abstractmethod, ABCMeta
from ais_bench.evaluate.log import logger
from ais_bench.infer.path_security_check import ms_open, MAX_SIZE_LIMITE_NORMAL_FILE

class BaseDataset(metaclass=ABCMeta):
    def __init__(self, dataset_name, dataset_path=None, shot=0) -> None:
        self.dataset_name = dataset_name
        self.shot = shot
        self.dataset_path = dataset_path
        self.load(dataset_path)

    def _download(self):
        os.chmod("download.sh", 0o755)
        result = subprocess.run(["./download.sh", self.dataset_name], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            current_script_path = os.path.realpath(__file__)
            parent_path = os.path.dirname(current_script_path)
            self.dataset_path = os.path.join(parent_path, self.dataset_name)
        else:
            logger.error("please download the dataset.")
            raise ValueError

    def _hash(file_path):
        hasher = hashlib.sha256()
        with ms_open(file_path, mode="rb", max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as file:
            for chunk in iter(lambda: file.read(4096), b''):
                hasher.update(chunk)
        hash_value = hasher.hexdigest()
        return hash_value

    def _check(self):
        parent_path = os.path.dirname(os.path.realpath(__file__))
        hash_json_path = os.path.join(parent_path, f"{self.dataset_name}_sha256.json")
        with ms_open(hash_json_path, max_size=MAX_SIZE_LIMITE_NORMAL_FILE) as file:
            hash_dict = json.load(file)
        dataset_dir = os.path.join(parent_path, self.dataset_name)
        for relative_path, true_hash in hash_dict.items():
            file_path = os.path.join(dataset_dir, relative_path)
            if (self._hash(file_path) != true_hash):
                logger.error("dataset verification failed: file hash value different")
                raise ValueError

    @abstractmethod
    def load(self, dataset_path):
        # to do : open file or files in dataset_path and save as pd.dataframe
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @abstractmethod
    def compute(self, data, measurement): # need to have a default measurement for every dataset
        raise NotImplementedError

    @abstractmethod
    def report(self, metrics):
        raise NotImplementedError
