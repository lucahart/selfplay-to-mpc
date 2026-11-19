"""Evaluate one saved student checkpoint closed-loop.

Usage:
    python scripts/distill/evaluate_checkpoint.py checkpoint=<pt> \
        student=mpc [mpc.<field>=<value> ...] [num_scenarios=1000] \
        [evaluator=validation_gigaflow] [device=cuda] [render_dir=<dir>] \
        [teacher_checkpoint=<pt>] [policy.<key>=<value> ...]

The student type and its settings must match the checkpoint (the state
dict loads strictly). Appends the metrics to final_eval.jsonl next to the
checkpoint and prints the main ones. evaluator= picks the [eval.<name>]
section (nuplan_multi_replan for nuPlan). With render_dir, the
evaluator's videos (one mp4 per rendered scenario and camera view) end up
directly in that folder; HTML replays (render_backend triage_html) stay in
<render_dir>/gif/<evaluator>/. teacher_checkpoint= and policy.* select
another teacher than TEACHER_CHECKPOINT, as in train.py.
"""

import json
import os
import shutil
import sys

import torch

from pufferlib.distill.bicycle import STEERING_RATE
from pufferlib.distill.env import TEACHER_CHECKPOINT, load_distill_args
from pufferlib.distill.loop import DEFAULT_EVALUATOR, DistillEvaluator
from pufferlib.distill.teacher import Teacher

# train.py sits next to this script (sys.path[0] when run as a script)
from train import STUDENT_MLP, STUDENT_TYPES, build_student

DEFAULT_NUM_SCENARIOS = 1000
FINAL_EVAL_LOG_NAME = "final_eval.jsonl"
TUBE_METRIC_PREFIX = "tube/"
PRINTED_METRICS = (
    "num_goals_reached",
    "collision_rate",
    "offroad_rate",
    "dnf_rate",
    "avg_speed_per_agent",
)


def parse_arguments(
        argv: list[str],
    ) -> dict:
    """
    Read checkpoint=, student=, <type>.<field>=, num_scenarios=,
    evaluator=, device=, render_dir=, teacher_checkpoint= and policy.*
    arguments.

    Args:
        argv (list[str]): command-line arguments after the script name.

    Returns:
        dict: checkpoint (str), student (str), student_strings
            (dict[str, str]), num_scenarios (int), evaluator (str),
            device (str), render_dir (str | None), teacher_checkpoint
            (str) and policy_overrides (list[str]).
    """
    settings = {
        "checkpoint": None,
        "student": STUDENT_MLP,
        "student_strings": {},
        "num_scenarios": DEFAULT_NUM_SCENARIOS,
        "evaluator": DEFAULT_EVALUATOR,
        "device": "cuda",
        "render_dir": None,
        "teacher_checkpoint": TEACHER_CHECKPOINT,
        "policy_overrides": [],
    }
    for argument in argv:
        key, value = argument.split("=", 1)
        prefix = key.split(".", 1)[0]
        if "." in key and prefix in STUDENT_TYPES:
            settings["student_strings"][key.split(".", 1)[1]] = value
        elif prefix == "policy" and "." in key:
            settings["policy_overrides"].append(argument)
        elif key == "num_scenarios":
            settings["num_scenarios"] = int(value)
        elif key == "render_dir":
            # The render pass changes the working directory
            settings[key] = os.path.abspath(value)
        elif key in ("checkpoint", "student", "evaluator", "device",
                     "teacher_checkpoint"):
            settings[key] = value
        else:
            raise ValueError(f"unknown key '{key}'")
    if settings["checkpoint"] is None:
        raise ValueError("checkpoint=<path> is required")
    if not os.path.isfile(settings["checkpoint"]):
        raise FileNotFoundError(settings["checkpoint"])
    return settings


def move_videos_up(
        render_dir: str,
        evaluator_name: str,
    ) -> None:
    """
    The evaluator writes videos to <render_dir>/mp4/<evaluator_name>/;
    move them into render_dir and remove the emptied subfolders.

    Args:
        render_dir (str): requested video folder.
        evaluator_name (str): evaluator section name.
    """
    written_dir = os.path.join(render_dir, "mp4", evaluator_name)
    video_names = sorted(
        name for name in os.listdir(written_dir) if name.endswith(".mp4")
    )
    if not video_names:
        raise RuntimeError(f"no videos were written to {written_dir}")
    for name in video_names:
        shutil.move(os.path.join(written_dir, name), render_dir)
    os.rmdir(written_dir)
    os.rmdir(os.path.dirname(written_dir))
    print(f"{len(video_names)} videos in {render_dir}")


def main() -> None:
    """
    Load the checkpoint into a student, evaluate it, log and print.
    """
    settings = parse_arguments(sys.argv[1:])
    device = settings["device"]
    args = load_distill_args(
        (f"train.device={device}", *settings["policy_overrides"])
    )
    teacher = Teacher(settings["teacher_checkpoint"], args, device)
    student = build_student(
        settings["student"], teacher, settings["student_strings"]
    ).to(device)
    checkpoint = torch.load(
        settings["checkpoint"], map_location=device, weights_only=False
    )
    student.load_state_dict(checkpoint["student"])
    labels_seen = int(checkpoint["labels_seen"])
    evaluator = DistillEvaluator(
        args,
        evaluator_name=settings["evaluator"],
        num_scenarios=settings["num_scenarios"],
    )
    render_dir = settings["render_dir"]
    metrics = evaluator.evaluate(
        student, render_results_dir=render_dir, global_step=labels_seen
    )
    mp4_dir = os.path.join(render_dir or "", "mp4", evaluator.evaluator_name)
    if render_dir is not None and os.path.isdir(mp4_dir):
        move_videos_up(render_dir, evaluator.evaluator_name)
    elif render_dir is not None:
        html_dir = os.path.join(render_dir, "gif", evaluator.evaluator_name)
        print(f"HTML replays in {html_dir}")

    row = {
        "checkpoint": os.path.abspath(settings["checkpoint"]),
        "student": settings["student"],
        "student_settings": settings["student_strings"],
        "teacher_checkpoint": os.path.abspath(
            settings["teacher_checkpoint"]
        ),
        "steering_rate_bound": float(student.control_bound[STEERING_RATE]),
        "labels_seen": labels_seen,
        "render_dir": render_dir,
        "num_scenarios": settings["num_scenarios"],
        "evaluator": settings["evaluator"],
        **metrics,
    }
    log_path = os.path.join(
        os.path.dirname(settings["checkpoint"]), FINAL_EVAL_LOG_NAME
    )
    with open(log_path, "a") as log_file:
        log_file.write(json.dumps(row) + "\n")
    print(f"\n{os.path.basename(settings['checkpoint'])} "
          f"({row['labels_seen']:,} labels, {row['num_scenarios']} "
          f"{row['evaluator']} scenarios)")
    for name in PRINTED_METRICS:
        print(f"  {name:>22} {metrics[name]:.4f}")
    # Only students with a closed-loop tube report them
    for name in sorted(metrics):
        if name.startswith(TUBE_METRIC_PREFIX):
            print(f"  {name:>22} {metrics[name]:.4f}")


if __name__ == "__main__":
    main()
