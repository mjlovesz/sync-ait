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

from ais_bench.evaluate.measurement.measurement import AccuracyMeasurement, EditDistanceMeasurement
from ais_bench.evaluate.log import logger

measurement_switch = {
    "accuracy": AccuracyMeasurement,
    "edit-distance": EditDistanceMeasurement
}


class MeasurementFactory():
    def get(self, measurement):
        if measurement_switch.get(measurement.strip()) is not None:
            return measurement_switch.get(measurement.strip())
        else:
            logger.error(f"Measurement {measurement} is not supported."
                         f"Currently only {', '.join(list(measurement_switch.keys()))} are supported.")
            raise ValueError