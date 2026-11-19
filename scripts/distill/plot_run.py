"""Plot a distillation run: training loss and closed-loop evaluations
against teacher labels seen.

Usage:
    python scripts/distill/plot_run.py run_dir=<dir> \
        [teacher=experiments/distill/teacher_check_1000.json] \
        [label="MLP student (BC)"] [scenarios=1000]

Writes <run_dir>/learning_curve.png and <run_dir>/eval_curve.png (plus
<run_dir>/probe_curve.png when the run logged forgetting-probe losses) and
prints the evaluation table.
"""

import json
import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pufferlib.distill.env import REPO_ROOT  # noqa: E402

DEFAULT_TEACHER_PATH = os.path.join(
    REPO_ROOT, "experiments", "distill", "teacher_check_1000.json"
)
# Teacher reference: its own (discrete) actions on the same evaluator
TEACHER_MODE = "discrete"
EVAL_PANELS = (
    ("num_goals_reached", "Goals per agent"),
    ("collision_rate", "Collision rate"),
    ("offroad_rate", "Off-road rate"),
)
DEFAULT_LABEL = "MLP student (BC)"
DEFAULT_SCENARIOS = "1000"
LOSS_SMOOTHING_ITERATIONS = 50
PROBE_PREFIX = "probe/"
LOSS_TICKS = (0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5)
LABELS_PER_MILLION = 1e6
FIGURE_DPI = 150

# Light-mode chart tokens (dataviz reference palette)
SURFACE = "#fcfcfb"
SERIES_BLUE = "#2a78d6"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def read_jsonl(
        path: str,
    ) -> list[dict]:
    """
    Args:
        path (str): JSON-lines file.

    Returns:
        list[dict]: one dict per line.
    """
    with open(path) as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]


def style_axes(
        axes: plt.Axes,
        title: str,
    ) -> None:
    """
    Recessive grid and axes, title in primary ink.

    Args:
        axes (plt.Axes): axes to style.
        title (str): panel title.
    """
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=PRIMARY_INK, fontsize=11, loc="left")
    axes.grid(True, color=GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(BASELINE)
    axes.tick_params(colors=MUTED_INK, labelsize=9, length=0)
    axes.set_xlabel("Teacher labels seen (millions)", color=SECONDARY_INK,
                    fontsize=9)


def plot_loss(
        log_rows: list[dict],
        label: str,
        output_path: str,
    ) -> None:
    """
    Training loss per iteration (faint) and its moving average.

    Args:
        log_rows (list[dict]): rows of log.jsonl.
        label (str): student name for the title.
        output_path (str): PNG path.
    """
    labels = np.array([row["labels_seen"] for row in log_rows])
    labels = labels / LABELS_PER_MILLION
    loss = np.array([row["loss"] for row in log_rows])
    window = min(LOSS_SMOOTHING_ITERATIONS, len(loss))
    smoothed = np.convolve(loss, np.ones(window) / window, mode="valid")

    figure, axes = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    axes.plot(labels, loss, color=SERIES_BLUE, alpha=0.2, linewidth=1)
    axes.plot(labels[window - 1:], smoothed, color=SERIES_BLUE, linewidth=2)
    axes.set_yscale("log")
    # Fixed readable ticks within the data range, plain decimals
    low, high = loss.min(), loss.max()
    ticks = [tick for tick in LOSS_TICKS if low * 0.9 <= tick <= high * 1.1]
    axes.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axes.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    style_axes(axes, f"{label}, training loss (normalized MSE)")
    axes.text(
        labels[-1], smoothed[-1], f"  {smoothed[-1]:.4f}",
        color=PRIMARY_INK, fontsize=9, va="center",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(figure)


def plot_probes(
        log_rows: list[dict],
        label: str,
        output_path: str,
    ) -> None:
    """
    Loss on every frozen forgetting-probe set, per iteration. A curve that
    turns upward after its set was frozen is forgetting.

    Args:
        log_rows (list[dict]): rows of log.jsonl.
        label (str): student name for the title.
        output_path (str): PNG path.
    """
    names = sorted({
        key for row in log_rows for key in row if key.startswith(PROBE_PREFIX)
    })
    labels = np.array([row["labels_seen"] for row in log_rows])
    labels = labels / LABELS_PER_MILLION
    figure, axes = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    colors = matplotlib.colormaps["viridis"](np.linspace(0.1, 0.85, len(names)))
    for name, color in zip(names, colors):
        loss = np.array([row.get(name, np.nan) for row in log_rows])
        axes.plot(labels, loss, color=color, linewidth=1.5)
        last = np.flatnonzero(~np.isnan(loss))[-1]
        axes.text(
            labels[last], loss[last], f"  {name[len(PROBE_PREFIX):]}",
            color=PRIMARY_INK, fontsize=8, va="center",
        )
    axes.set_yscale("log")
    axes.yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0))
    )
    axes.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter("%.2f")
    )
    style_axes(axes, f"{label}, forgetting probe (normalized MSE)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(figure)


def plot_evaluations(
        eval_rows: list[dict],
        teacher_metrics: dict | None,
        title: str,
        output_path: str,
    ) -> None:
    """
    One small panel per metric: the student at each evaluation, and the
    teacher as a dashed reference line.

    Args:
        eval_rows (list[dict]): rows of eval.jsonl.
        teacher_metrics (dict | None): teacher's metrics on the same
            evaluator, or None.
        title (str): figure title.
        output_path (str): PNG path.
    """
    labels = np.array([row["labels_seen"] for row in eval_rows])
    labels = labels / LABELS_PER_MILLION
    figure, panels = plt.subplots(
        1, len(EVAL_PANELS), figsize=(12, 3.6), facecolor=SURFACE
    )
    for axes, (metric, panel_title) in zip(panels, EVAL_PANELS):
        values = np.array([row[metric] for row in eval_rows])
        axes.plot(labels, values, color=SERIES_BLUE, linewidth=2,
                  marker="o", markersize=6)
        axes.text(labels[-1], values[-1], f"  {values[-1]:.3f}",
                  color=PRIMARY_INK, fontsize=9, va="center")
        if teacher_metrics is not None:
            teacher_value = teacher_metrics[metric]
            axes.axhline(teacher_value, color=SECONDARY_INK, linewidth=1.5,
                         linestyle=(0, (4, 3)))
            # Offset so the label sits above the dashed line, not on it
            axes.annotate(
                f"teacher {teacher_value:.3f}",
                (labels[0], teacher_value),
                xytext=(0, 4),
                textcoords="offset points",
                color=SECONDARY_INK,
                fontsize=9,
                va="bottom",
            )
        style_axes(axes, panel_title)
        axes.set_ylim(bottom=0)
    figure.suptitle(title, color=PRIMARY_INK, fontsize=12, x=0.01,
                    ha="left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    """
    Read the run's logs, write both figures, print the evaluation table.
    """
    settings = {
        "run_dir": None,
        "teacher": DEFAULT_TEACHER_PATH,
        "label": DEFAULT_LABEL,
        "scenarios": DEFAULT_SCENARIOS,
    }
    for argument in sys.argv[1:]:
        key, value = argument.split("=", 1)
        if key not in settings:
            raise ValueError(f"unknown key '{key}'")
        settings[key] = value
    run_dir = settings["run_dir"]
    if run_dir is None:
        raise ValueError("run_dir=<dir> is required")
    log_rows = read_jsonl(os.path.join(run_dir, "log.jsonl"))
    eval_rows = read_jsonl(os.path.join(run_dir, "eval.jsonl"))
    teacher_metrics = None
    if os.path.isfile(settings["teacher"]):
        with open(settings["teacher"]) as teacher_file:
            teacher_metrics = json.load(teacher_file)[TEACHER_MODE]

    label = settings["label"]
    plot_loss(log_rows, label, os.path.join(run_dir, "learning_curve.png"))
    if any(key.startswith(PROBE_PREFIX) for row in log_rows for key in row):
        plot_probes(log_rows, label, os.path.join(run_dir, "probe_curve.png"))
    eval_title = (
        f"{label}, validation_gigaflow, {settings['scenarios']} scenarios"
    )
    plot_evaluations(
        eval_rows,
        teacher_metrics,
        eval_title,
        os.path.join(run_dir, "eval_curve.png"),
    )
    print(f"{'labels (M)':>10} {'goals':>7} {'collision':>9} {'offroad':>8}")
    if teacher_metrics is not None:
        print(
            f"{'teacher':>10} {teacher_metrics['num_goals_reached']:>7.3f} "
            f"{teacher_metrics['collision_rate']:>9.4f} "
            f"{teacher_metrics['offroad_rate']:>8.4f}"
        )
    for row in eval_rows:
        print(
            f"{row['labels_seen'] / LABELS_PER_MILLION:>10.1f} "
            f"{row['num_goals_reached']:>7.3f} "
            f"{row['collision_rate']:>9.4f} {row['offroad_rate']:>8.4f}"
        )


if __name__ == "__main__":
    main()
