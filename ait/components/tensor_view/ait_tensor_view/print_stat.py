import torch
from tabulate import tabulate


def print_stat(tensor: torch.Tensor):
    table = [
        ["min", "max", "mean", "std", "var"],
        [tensor.min(), tensor.max(), tensor.mean(), tensor.std(), tensor.var()]
    ]

    print(tabulate(table, tablefmt="grid"))
