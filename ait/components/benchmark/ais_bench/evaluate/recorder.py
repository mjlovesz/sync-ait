import pandas as pd

class Recorder():
    def __init__(self, name) -> None:
        self.name = name
        self.records = None
        self.metrics = None
        self.children = dict()

    def record(self, index : list, entry_dict : dict):
        '''
        all the index should be of the same length
        '''
        if index == []:
            if self.records is None:
                self.records = pd.DataFrame(columns=entry_dict.keys())
            df_dict = pd.DataFrame([entry_dict])
            self.records = pd.concat([self.records, df_dict], ignore_index=True)
            return

        current_index = index[0]
        if current_index not in self.children:
            self.children[current_index] = Recorder(current_index)
        self.children.get(current_index).record(index[1:], entry_dict)

    def read(self, index):
        if index == []:
            return self.records
        current_index = index[0]
        return self.children.get(current_index).read(index[1:])

    def statistics(self, func_compute = None):
        if self.metrics is not None:
            return self.metrics
        if func_compute is None:
            print("Record.statistics failed: function to compute metrics missing")
            raise Exception

        if not self.children:
            data = self.records
        else:
            data = []
            for child in self.children:
                data.append(child.statistics())

        self.metrics = func_compute(data)
        return self.metrics


    def report(self, func_report = None):
        if func_report is None:
            print("Record.report failed: function to report metrics missing")
            raise Exception
        func_report(self.metrics)






    # def build(self, index):
    #     if index == []:
    #         return
    #     if index[0] not in self.records:
    #         self.records[index[0]] = Recorder(self.rank)

    #     self.records.get(index[0]).build(index[1:])
