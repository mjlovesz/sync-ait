import re

import torch

SLICE_PATTERN = re.compile(r"\.\.\.|\d+|(\d+)?:(\d+)?(:?\d+)?")


class Operation:
    def process(self, prev: torch.Tensor) -> torch.Tensor:
        pass


def convert_slice(s: str) -> slice:
    parts = [n.strip() for n in s.split(":")]
    size = len(parts)

    if size == 2:
        start, end = parts
        start = int(start) if start else None
        end = int(end) if end else None
        return slice(start, end, None)
    elif size == 3:
        start, end, step = parts
        start = int(start) if start else None
        end = int(end) if end else None
        step = int(step) if step else None
        return slice(start, end, step)
    else:
        raise SyntaxError(f"'{s}' is not valid slice string")


class SliceOperation(Operation):
    def __init__(self, slice_str: str):
        content = slice_str[1:-1].split(",")

        parts = []

        for c in content:
            c = c.strip()
            if c != "":
                if not SLICE_PATTERN.match(c):
                    raise SyntaxError(f"slice string is invalid: '{slice_str}', part: '{c}'")
                parts.append(c)

        if len(parts) == 0:
            raise SyntaxError("slice string can not be empty")

        self.slice_raw = slice_str
        self.parts = parts

    def process(self, prev: torch.Tensor) -> torch.Tensor:
        size = len(self.parts)
        shape = prev.shape
        shape_size = len(shape)

        if size == 1:
            src = self.parts[0]
            if ":" in src:
                return prev[convert_slice(src)]
            elif src == "...":
                return prev
            else:
                index = int(src)
                dim_size0 = shape[0]
                if index >= dim_size0 or -1 * index > dim_size0:
                    raise ValueError(f"Index out of range , dim0_size is {dim_size0} while index={index}")
                return prev[index]
        else:
            if size > shape_size:
                raise ValueError(f"number of dimensions[{self.slice_raw}] is bigger than the shape size[{shape_size}]")
            slices = []

            for i, part in enumerate(self.parts):
                if ":" in part:
                    slices.append(convert_slice(part))
                elif part == "...":
                    slices.append(Ellipsis)
                else:
                    index = int(part)
                    dim_size = shape[i]
                    if index >= dim_size or -1 * index > dim_size:
                        raise ValueError(f"Index out of range, dim[{i}]_size is {dim_size} while index={index}")
                    slices.append(index)

            return prev[slices]


class PermuteOperation(Operation):
    def __init__(self, permute_str: str):
        parts = [int(n.strip()) for n in permute_str[1:-1].split(",") if n != ""]
        self.permute_raw = permute_str
        self.parts = parts

    def process(self, prev: torch.Tensor) -> torch.Tensor:
        shape = prev.shape
        n = len(shape)

        if len(self.parts) != len(shape):
            raise ValueError(f"'{self.permute_raw}' has no enough dimension")

        self.check_permute_elements()
        self.check_permute_range(n)

        return prev.permute(self.parts)

    def check_permute_elements(self):
        if not all(isinstance(x, int) for x in self.parts):
            raise ValueError(f"All elements in {self.permute_raw} should be integer, but got {self.parts}")
        if len(set(self.parts)) != len(self.parts):
            raise ValueError(f"{self.permute_raw} contains duplicate dimensions!")

    def check_permute_range(self, n: int):
        if not all(0 <= x < n for x in set(self.parts)):
            raise ValueError(f"{self.permute_raw}: not all dimensions are between 0 and {n - 1}")
