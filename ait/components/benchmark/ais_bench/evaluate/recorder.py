import pandas as pd

class Recorder():
    def __init__(self) -> None:
        self.records = dict()

    def record(self, index : list, entry_dict : dict):
        '''
        all the index should be of the same length
        '''
        if index == []:
            if self.records is dict():
                self.records = pd.DataFrame(columns=entry_dict.keys())
            df_dict = pd.DataFrame([entry_dict])
            self.records = pd.concat([self.records, df_dict], ignore_index=True)
            return

        if index[0] not in self.records:
            self.records[index[0]] = Recorder()
        self.records.get(index[0]).record(index[1:], entry_dict)

    def read(self, index):
        if index == []:
            return self.records

        return self.records.get(index[0]).read(index[1:])

    # def build(self, index):
    #     if index == []:
    #         return
    #     if index[0] not in self.records:
    #         self.records[index[0]] = Recorder(self.rank)

    #     self.records.get(index[0]).build(index[1:])
