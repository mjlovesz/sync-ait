from abc import abstractmethod, ABCMeta

class BaseMeasurement(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, data, **kwargs):
        raise NotImplementedError

class AccuracyMeasurement(BaseMeasurement):
    def __call__(self, data, ground_truth_index, answer_index):
        out_dict = dict()
        out_dict["total amount"] = 0
        out_dict["correct amount"] = 0
        if isinstance(data, list):
            for metrics in data:
                out_dict["total amount"] += metrics.get("total amount")
                out_dict["correct amount"] += metrics.get("correct amount")
        else:
            out_dict["total amount"] = data.shape[0]
            out_dict["correct amount"] = len(data[data[ground_truth_index] == data[answer_index]])
        if out_dict["total amount"] == 0:
            out_dict["accuracy"] = 0
        else:
            out_dict["accuracy"] = out_dict["correct amount"] / out_dict["total amount"]

        return out_dict