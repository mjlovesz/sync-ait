import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("../../")) ##保证amit入口和debug/compare入口
from profile.benchmark.main_cli import benchmark_cli_enter


if __name__ == "__main__":
    benchmark_cli_enter()
