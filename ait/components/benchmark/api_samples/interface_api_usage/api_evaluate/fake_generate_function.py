from ais_bench.evaluate.interface import Evaluator


def generate_func(prompt):
    return "A"

evaluator = Evaluator(generate_func, "ceval", shot=5)
evaluator.evaluate()