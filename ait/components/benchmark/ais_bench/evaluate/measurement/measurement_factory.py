from ais_bench.evaluate.measurement.measurement import AccuracyMeasurement, EditDistanceMeasurement

measurement_switch = {
    "accuracy": AccuracyMeasurement,
    "edit-distance": EditDistanceMeasurement
}

class MeasurementFactory():
    def get(self, measurement):
        if measurement_switch.get(measurement.strip()) is not None:
            return measurement_switch.get(measurement.strip())
        else:
            print(f"Measurement {measurement} is not supported."
                  f"Currently only {', '.join(list(measurement_switch.keys()))} are supported.")
            raise ValueError