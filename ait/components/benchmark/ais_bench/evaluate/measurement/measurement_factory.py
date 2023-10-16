from ais_bench.evaluate.measurement.measurement import AccuracyMeasurement

class MeasurementFactory():
    def get_dataset(self, measurement):
        if measurement == "accuracy":
            return AccuracyMeasurement
        else:
            print(f"measurement {measurement} is not supported")
            raise ValueError