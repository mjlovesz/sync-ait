import pandas as pd
from ais_bench.evaluate.log import logger

class Recorder():
    def __init__(self, name = "default", rank = 0) -> None:
        self.name = name
        self.records = None
        self.metrics = None
        self.children = dict()
        self.rank = rank

    def record(self, index : list, entry_dict : dict):
        '''
        all the index should be of the same length
        '''
        if self.rank != 0:
            return
        if index == []:
            if self.records is None:
                self.records = pd.DataFrame(columns=entry_dict.keys())
            df_dict = pd.DataFrame([entry_dict])
            self.records = pd.concat([self.records, df_dict], ignore_index=True)
            return

        current_index = index[0]
        if current_index not in self.children:
            self.children[current_index] = Recorder(current_index, self.rank)
        self.children.get(current_index).record(index[1:], entry_dict)

    def read(self, index):
        if self.rank != 0:
            return
        if index == []:
            return self.records
        current_index = index[0]
        return self.children.get(current_index).read(index[1:])

    def statistics(self, func_compute = None, measurement = None):
        if self.rank != 0:
            return
        if self.metrics is not None:
            return self.metrics
        if func_compute is None:
            logger.error("Record.statistics failed: function to compute metrics missing")
            raise Exception

        if not self.children:
            data = self.records
        else:
            data = []
            for child in self.children.values():
                data.append(child.statistics(func_compute, measurement))

        if measurement is None:
            # use default measurement
            self.metrics = func_compute(data)
        else:
            self.metrics = func_compute(data, measurement)
        return self.metrics


    def report(self, func_report = None):
        if self.rank != 0:
            return
        if func_report is None:
            logger.error("Record.report failed: function to report metrics missing")
            raise Exception
        func_report(self.metrics)






    # def build(self, index):
    #     if index == []:
    #         return
    #     if index[0] not in self.records:
    #         self.records[index[0]] = Recorder(self.rank)

    #     self.records.get(index[0]).build(index[1:])
