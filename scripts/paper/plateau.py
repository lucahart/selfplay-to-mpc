"""Stopping rule and checkpoint choice of the paper's DAgger runs.

Usage:
    python scripts/paper/plateau.py stop <eval.jsonl>  prints stop/continue
    python scripts/paper/plateau.py best <eval.jsonl>  prints labels_seen

Both read the 250-scenario CARLA evaluations a run writes every 10M
labels. The score of an evaluation is its collision plus off-road rate
(lower is better). A run stops after PATIENCE_EVALS evaluations in a row
that neither lower the best score nor raise the best goals reached by more
than GOALS_TOLERANCE. The chosen checkpoint is the one with the lowest
score. The evaluation of the untrained student (0 labels) is ignored.
"""

import json
import sys

PATIENCE_EVALS = 2
# About the 95% binomial half-width of goals reached on 250 scenarios
GOALS_TOLERANCE = 0.03


def load_evals(
        path: str,
    ) -> list[dict]:
    """
    Read the evaluations of trained students of a run.

    Args:
        path (str): the run's eval.jsonl.

    Returns:
        list[dict]: rows with labels_seen > 0, in label order.
    """
    with open(path) as log:
        evals = [json.loads(line) for line in log if line.strip()]
    evals = [row for row in evals if row["labels_seen"] > 0]
    return sorted(evals, key=lambda row: row["labels_seen"])


def score(
        row: dict,
    ) -> float:
    """
    Collision plus off-road rate of one evaluation.

    Args:
        row (dict): one eval.jsonl row.

    Returns:
        float: the score, lower is better.
    """
    return row["collision_rate"] + row["offroad_rate"]


def should_stop(
        evals: list[dict],
    ) -> bool:
    """
    Whether the last PATIENCE_EVALS evaluations all failed to improve.

    Args:
        evals (list[dict]): evaluations in label order.

    Returns:
        bool: True to stop the run.
    """
    best_score = float("inf")
    best_goals = float("-inf")
    evals_without_improvement = 0
    for row in evals:
        improved = (
            score(row) < best_score
            or row["num_goals_reached"] > best_goals + GOALS_TOLERANCE
        )
        evals_without_improvement = 0 if improved else (
            evals_without_improvement + 1
        )
        best_score = min(best_score, score(row))
        best_goals = max(best_goals, row["num_goals_reached"])
    return evals_without_improvement >= PATIENCE_EVALS


def best_labels(
        evals: list[dict],
    ) -> int:
    """
    labels_seen of the evaluation with the lowest score.

    Args:
        evals (list[dict]): evaluations, at least one.

    Returns:
        int: labels_seen of the chosen checkpoint.
    """
    return min(evals, key=score)["labels_seen"]


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    run_evals = load_evals(path)
    if mode == "stop":
        print("stop" if should_stop(run_evals) else "continue")
    elif mode == "best":
        print(best_labels(run_evals))
    else:
        raise ValueError(f"unknown mode {mode}")
