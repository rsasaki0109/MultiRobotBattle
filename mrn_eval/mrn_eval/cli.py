"""Command line entry point for evaluation helpers."""

import argparse

from mrn_eval.metrics import ate_2d


def main() -> None:
    parser = argparse.ArgumentParser(prog="mrn_eval")
    parser.add_argument("--demo", action="store_true", help="print a deterministic demo ATE")
    args = parser.parse_args()

    if args.demo:
        value = ate_2d([(0.0, 0.0), (1.1, 0.0)], [(0.0, 0.0), (1.0, 0.0)])
        print(f"ate_2d={value:.6f}")
    else:
        parser.print_help()
