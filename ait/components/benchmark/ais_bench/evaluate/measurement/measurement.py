# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


class EditDistanceMeasurement(BaseMeasurement):
    def _editing_distance(self, word1, word2):
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        for i in range(len(word1) + 1):
            dp[i][0] = i
        for j in range(len(word2) + 1):
            dp[0][j] = j
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[-1][-1]

    def __call__(self, data, ground_truth_index, answer_index):
        out_dict = dict()
        out_dict["total amount"] = 0
        distance_sum = 0
        if isinstance(data, list):
            for metrics in data:
                out_dict["total amount"] += metrics.get("total amount")
                distance_sum += metrics.get("total amount") * metrics.get("editing distance")
        else:
            out_dict["total amount"] = data.shape[0]
            for _, row in data.iterrows():
                distance_sum += self._editing_distance(row[ground_truth_index], row[answer_index])
        if out_dict["total amount"] == 0:
            out_dict["editing distance"] = 0
        else:
            out_dict["editing distance"] = distance_sum / out_dict["total amount"]

        return out_dict
