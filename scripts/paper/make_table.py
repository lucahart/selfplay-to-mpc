"""Table 1 of the paper: failure rate (collision plus off-road rate per
vehicle) of the teachers and students on the three test sets, as the mean
and standard error over seeds.

Usage:
    python scripts/paper/make_table.py [output_root]

Reads the runs and teacher evaluations that train_student.sh,
evaluate_student.sh and evaluate_teacher.sh wrote under output_root
(default experiments/paper) and prints a Markdown table.
"""

import json
import math
import os
import statistics
import sys

# plateau.py sits next to this script (sys.path[0] when run as a script)
from plateau import best_labels, load_evals

DEFAULT_OUTPUT_ROOT = "experiments/paper"
SIZES = ("small", "large")
STUDENTS = ("mlp", "unconstrained", "constrained")
SEEDS = range(1, 10)
TEST_SETS = (
    ("CARLA", "validation_gigaflow"),
    ("nuPlan", "nuplan_multi_replan"),
    ("Waymo", "womd_multi"),
)
FINAL_EVAL_LOG_NAME = "final_eval.jsonl"
EVAL_LOG_NAME = "eval.jsonl"


def failure_rate(
        metrics: dict,
    ) -> float:
    """
    Collision plus off-road rate of one evaluation.

    Args:
        metrics (dict): evaluator metrics.

    Returns:
        float: the failure rate per vehicle.
    """
    return metrics["collision_rate"] + metrics["offroad_rate"]


def student_rates(
        run_dir: str,
    ) -> dict[str, float]:
    """
    Failure rate per evaluator of a run's chosen checkpoint (plateau.py),
    from final_eval.jsonl.

    Args:
        run_dir (str): the run folder.

    Returns:
        dict[str, float]: evaluator -> failure rate; empty if the run was
            not evaluated.
    """
    path = os.path.join(run_dir, FINAL_EVAL_LOG_NAME)
    if not os.path.isfile(path):
        return {}
    labels = best_labels(load_evals(os.path.join(run_dir, EVAL_LOG_NAME)))
    rates = {}
    with open(path) as log:
        for line in log:
            row = json.loads(line)
            if row["labels_seen"] == labels:
                rates[row["evaluator"]] = failure_rate(row)
    return rates


def teacher_rates(
        teacher_dir: str,
    ) -> dict[str, float]:
    """
    Failure rate per evaluator of a teacher with its discrete actions,
    from the result line check_teacher.py prints last.

    Args:
        teacher_dir (str): folder of evaluate_teacher.sh's outputs.

    Returns:
        dict[str, float]: evaluator -> failure rate.
    """
    rates = {}
    for _, evaluator in TEST_SETS:
        path = os.path.join(teacher_dir, f"{evaluator}.out")
        if not os.path.isfile(path):
            continue
        with open(path) as output:
            result_lines = [
                line for line in output if line.startswith("{")
            ]
        if result_lines:
            results = json.loads(result_lines[-1])
            rates[evaluator] = failure_rate(results["discrete"])
    return rates


def mean_and_error(
        values: list[float],
    ) -> str:
    """
    Mean and standard error, formatted for the table.

    Args:
        values (list[float]): one value per seed.

    Returns:
        str: "mean ± se (n)", or "-" without values.
    """
    if not values:
        return "-"
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.3f} (1)"
    error = statistics.stdev(values) / math.sqrt(len(values))
    return f"{mean:.3f} ± {error:.3f} ({len(values)})"


def main() -> None:
    """
    Print the table.
    """
    output_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_ROOT
    header = " | ".join(name for name, _ in TEST_SETS)
    print(f"| teacher | policy | {header} |")
    print("|---|---|" + "---|" * len(TEST_SETS))
    for size in SIZES:
        teacher = teacher_rates(os.path.join(output_root, f"teacher_{size}"))
        cells = [
            f"{teacher[evaluator]:.3f}" if evaluator in teacher else "-"
            for _, evaluator in TEST_SETS
        ]
        print(f"| {size} | teacher | {' | '.join(cells)} |")
        for student in STUDENTS:
            seed_rates = [
                student_rates(
                    os.path.join(output_root, f"{size}_{student}_s{seed}")
                )
                for seed in SEEDS
            ]
            cells = [
                mean_and_error([
                    rates[evaluator] for rates in seed_rates
                    if evaluator in rates
                ])
                for _, evaluator in TEST_SETS
            ]
            print(f"| {size} | {student} | {' | '.join(cells)} |")


if __name__ == "__main__":
    main()
