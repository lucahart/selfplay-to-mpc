"""Teacher check: the teacher must drive as well through the continuous
action mode (decoded labels) as through its own discrete actions.

Usage:
    python scripts/distill/check_teacher.py [num_scenarios=1000] \
        [evaluator=validation_gigaflow] [device=cuda] [render_dir=<dir>] \
        [checkpoint=<pt>] [env.<key>=<value> ...] \
        [policy.<key>=<value> ...] [[+]eval.<key>=<value> ...]

Runs the evaluator (default validation_gigaflow; nuplan_multi for the
nuPlan OOD test) once per action mode; the discrete run is the teacher's
own number. The evaluator is a fixed benchmark (fixed reset seed), so
repeated runs are identical; agreement is judged per rate with a binomial
z-score over the evaluated agents. The two modes diverge
chaotically per trajectory, so they are compared as samples, not
trajectories. With render_dir, the discrete run also renders, into
<render_dir>/mp4/<evaluator>/ (egl) or <render_dir>/gif/<evaluator>/
(triage_html). checkpoint= checks another teacher than TEACHER_CHECKPOINT,
env.<key>= adds env overrides (e.g. the flags it was trained with),
policy.<key>= sets its network (e.g. the 4x1024 teacher's sizes) and
eval.<key>= changes the evaluator sections (+eval. adds a key).
"""

import json
import math
import os
import sys

from pufferlib.distill.env import TEACHER_CHECKPOINT, load_distill_args
from pufferlib.distill.loop import DEFAULT_EVALUATOR, DistillEvaluator
from pufferlib.distill.teacher import Teacher

# 250 scenarios left a borderline z (2.9) that vanished at 1000
DEFAULT_NUM_SCENARIOS = 1000
# Hydra override prefixes passed through to the composed config
CONFIG_OVERRIDE_PREFIXES = ("env.", "+env.", "policy.", "eval.", "+eval.")
# Per-agent rates in [0, 1], compared with a binomial z-score
COMPARED_RATES = (
    "collision_rate",
    "offroad_rate",
    "red_light_violation_rate",
    "dnf_rate",
)
REPORTED_METRICS = ("num_goals_reached", "score") + COMPARED_RATES
MAX_ABS_Z = 3.0


def parse_arguments(
        argv: list[str],
    ) -> dict:
    """
    Read num_scenarios=, evaluator=, device=, render_dir=, checkpoint=
    and env./policy./eval. override arguments.

    Args:
        argv (list[str]): command-line arguments after the script name.

    Returns:
        dict: num_scenarios (int), evaluator (str), device (str),
            render_dir (str | None), checkpoint (str) and
            config_overrides (tuple[str, ...], Hydra overrides on top of
            the teacher's).
    """
    settings = {
        "num_scenarios": DEFAULT_NUM_SCENARIOS,
        "evaluator": DEFAULT_EVALUATOR,
        "device": "cuda",
        "render_dir": None,
        "checkpoint": TEACHER_CHECKPOINT,
        "config_overrides": (),
    }
    for argument in argv:
        key, value = argument.split("=", 1)
        if key.startswith(CONFIG_OVERRIDE_PREFIXES):
            settings["config_overrides"] += (argument,)
        elif key == "checkpoint":
            if not os.path.isfile(value):
                raise FileNotFoundError(value)
            settings[key] = value
        elif key == "num_scenarios":
            settings["num_scenarios"] = int(value)
        elif key in ("evaluator", "device"):
            settings[key] = value
        elif key == "render_dir":
            # The render pass changes the working directory
            settings[key] = os.path.abspath(value)
        else:
            raise ValueError(f"unknown key '{key}'")
    return settings


def rate_z_score(
        discrete_rate: float,
        continuous_rate: float,
        agent_count: float,
    ) -> float:
    """
    Two-sample binomial z-score of a per-agent rate.

    Args:
        discrete_rate (float): rate in the discrete run.
        continuous_rate (float): rate in the continuous run.
        agent_count (float): agents evaluated in each run.

    Returns:
        float: (continuous - discrete) / standard error; 0 if both are 0.
    """
    variance = (
        discrete_rate * (1 - discrete_rate)
        + continuous_rate * (1 - continuous_rate)
    ) / agent_count
    if variance == 0:
        return 0.0
    return (continuous_rate - discrete_rate) / math.sqrt(variance)


def main() -> None:
    """
    Evaluate the teacher in both action modes, print the comparison and
    exit nonzero if a rate differs by more than MAX_ABS_Z.
    """
    settings = parse_arguments(sys.argv[1:])
    args = load_distill_args(
        (f"train.device={settings['device']}",)
        + settings["config_overrides"]
    )
    teacher = Teacher(settings["checkpoint"], args, settings["device"])
    evaluator = DistillEvaluator(
        args,
        evaluator_name=settings["evaluator"],
        num_scenarios=settings["num_scenarios"],
    )
    results = {
        "discrete": evaluator.evaluate(
            teacher,
            action_type="discrete",
            render_results_dir=settings["render_dir"],
        ),
        "continuous": evaluator.evaluate(teacher, action_type="continuous"),
    }

    agent_count = results["discrete"]["num_agents_evaluated"]
    print(f"\n{'metric':>26} {'discrete':>10} {'continuous':>10} {'z':>6}")
    worst_z = 0.0
    for name in REPORTED_METRICS:
        discrete_value = results["discrete"][name]
        continuous_value = results["continuous"][name]
        z_text = ""
        if name in COMPARED_RATES:
            z = rate_z_score(discrete_value, continuous_value, agent_count)
            worst_z = max(worst_z, abs(z))
            z_text = f"{z:6.2f}"
        print(
            f"{name:>26} {discrete_value:>10.4f} "
            f"{continuous_value:>10.4f} {z_text:>6}"
        )
    passed = worst_z <= MAX_ABS_Z
    print(f"{settings['evaluator']}: agents per run: {agent_count:.0f}, "
          f"worst |z| {worst_z:.2f}, "
          f"{'PASS' if passed else 'FAIL'} (|z| <= {MAX_ABS_Z})")
    print(json.dumps(results))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
