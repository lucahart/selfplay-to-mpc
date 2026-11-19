"""Distill the PPO teacher into a student with online BC or DAgger.

Usage:
    python scripts/distill/train.py [key=value ...]

Keys are DistillConfig fields (pufferlib/distill/loop.py) plus:
    agent_count=1024            agents in the continuous env (split over
                                env_worker_count worker processes)
    student=mlp                 student type: mlp or mpc
    mlp.<field>=<value>         MLPStudentConfig fields (students.py)
    mpc.<field>=<value>         MPCStudentConfig fields (students.py)
    teacher_checkpoint=<path>   defaults to env.TEACHER_CHECKPOINT
    eval_num_scenarios=<n>      evaluator scenarios (default: section's)
    env.<key>=<value>           extra Hydra overrides of the env section
    policy.<key>=<value>        Hydra overrides of the teacher's network
                                (another teacher than TEACHER_CHECKPOINT)
Example (BC, MLP student):
    python scripts/distill/train.py iterations=200 \
        output_dir=experiments/distill/bc_mlp_seed1
"""

import dataclasses
import os
import sys

import torch

from pufferlib.distill.buffer import LabelBuffer
from pufferlib.distill.env import (
    NUPLAN_ENV_OVERRIDES,
    REPO_ROOT,
    TEACHER_CHECKPOINT,
    load_distill_args,
    make_env,
)
from pufferlib.distill.loop import DistillConfig, DistillEvaluator, distill
from pufferlib.distill.students import (
    MLPStudent,
    MLPStudentConfig,
    MPCStudent,
    MPCStudentConfig,
    Student,
)
from pufferlib.distill.teacher import Teacher
from pufferlib.distill.vec_env import VecDrive

STUDENT_MLP = "mlp"
STUDENT_MPC = "mpc"
STUDENT_TYPES = (STUDENT_MLP, STUDENT_MPC)
STUDENT_CONFIG_CLASSES = {
    STUDENT_MLP: MLPStudentConfig,
    STUDENT_MPC: MPCStudentConfig,
}
TRUE_STRINGS = ("1", "true", "yes")
FALSE_STRINGS = ("0", "false", "no")
SCRIPT_KEYS = ("agent_count", "student", "teacher_checkpoint",
               "eval_num_scenarios")
DEFAULT_AGENT_COUNT = 1024


def cast_field(
        config_class: type,
        key: str,
        text: str,
    ) -> int | float | str | bool:
    """
    Cast a command-line value to the type of a dataclass field.

    Args:
        config_class (type): dataclass with the field.
        key (str): field name.
        text (str): value as typed.

    Returns:
        int | float | str | bool: the cast value.
    """
    field_types = {
        field.name: field.type for field in dataclasses.fields(config_class)
    }
    if key not in field_types:
        raise ValueError(f"unknown key '{key}'")
    # Field types are the annotation strings or the types themselves
    type_name = getattr(field_types[key], "__name__", field_types[key])
    if type_name == "bool":
        if text.lower() not in TRUE_STRINGS + FALSE_STRINGS:
            raise ValueError(f"'{key}' expects true or false")
        return text.lower() in TRUE_STRINGS
    caster = {"int": int, "float": float, "str": str}[type_name]
    return caster(text)


def build_student(
        student_type: str,
        teacher: Teacher,
        student_strings: dict[str, str],
    ) -> Student:
    """
    Construct a student of the given type from command-line settings.

    Args:
        student_type (str): "mlp" or "mpc".
        teacher (Teacher): source of the backbone.
        student_strings (dict[str, str]): <field> -> value for the
            student's config dataclass.

    Returns:
        Student: the student (on the CPU).
    """
    if student_type not in STUDENT_TYPES:
        raise ValueError(f"student must be one of {STUDENT_TYPES}")
    config_class = STUDENT_CONFIG_CLASSES[student_type]
    config = config_class(
        **{
            key: cast_field(config_class, key, text)
            for key, text in student_strings.items()
        }
    )
    if student_type == STUDENT_MPC:
        return MPCStudent(teacher, config)
    return MLPStudent(teacher, config)


def parse_overrides(
        argv: list[str],
    ) -> tuple[DistillConfig, dict, tuple[str, ...]]:
    """
    Split key=value arguments into the run config, script settings
    (including the student's <type>.<field> settings) and env overrides,
    casting each value to its field's type.

    Args:
        argv (list[str]): command-line arguments after the script name.

    Returns:
        tuple[DistillConfig, dict, tuple[str, ...]]: run config, script
            settings, and Hydra env overrides.
    """
    config_fields = {field.name for field in dataclasses.fields(DistillConfig)}
    config_values, env_overrides = {}, []
    script_settings = {
        "agent_count": DEFAULT_AGENT_COUNT,
        "student": STUDENT_MLP,
        "teacher_checkpoint": TEACHER_CHECKPOINT,
        "eval_num_scenarios": None,
        "student_strings": {},
    }
    student_prefixes = tuple(f"{name}." for name in STUDENT_TYPES)
    for argument in argv:
        if "=" not in argument:
            raise ValueError(f"expected key=value, got '{argument}'")
        key, value = argument.split("=", 1)
        # policy.* selects another teacher's network (env.py
        # TEACHER_POLICY_OVERRIDES is the default teacher's)
        if key.startswith(("env.", "policy.")):
            env_overrides.append(argument)
        elif key.startswith(student_prefixes):
            student_type, field = key.split(".", 1)
            script_settings["student_strings"][(student_type, field)] = value
        elif key in config_fields:
            config_values[key] = cast_field(DistillConfig, key, value)
        elif key in SCRIPT_KEYS:
            integer_keys = ("agent_count", "eval_num_scenarios")
            script_settings[key] = int(value) if key in integer_keys else value
        else:
            raise ValueError(f"unknown key '{key}'")
    student_type = script_settings["student"]
    if student_type not in STUDENT_TYPES:
        raise ValueError(f"student must be one of {STUDENT_TYPES}")
    # Settings for another student type are a mistake, not a no-op
    for other_type, field in script_settings["student_strings"]:
        if other_type != student_type:
            raise ValueError(
                f"'{other_type}.{field}' needs student={other_type}"
            )
    script_settings["student_strings"] = {
        field: value
        for (_, field), value in script_settings["student_strings"].items()
    }
    if not config_values.get("output_dir"):
        config_values["output_dir"] = os.path.join(
            REPO_ROOT,
            "experiments",
            "distill",
            f"{script_settings['student']}_seed{config_values.get('seed', 1)}",
        )
    return DistillConfig(**config_values), script_settings, tuple(
        env_overrides
    )


def main() -> None:
    """
    Build env, teacher, student and buffer from the arguments; run the
    loop.
    """
    config, settings, env_overrides = parse_overrides(sys.argv[1:])
    torch.manual_seed(config.seed)
    if config.env_worker_count > 1:
        args = load_distill_args(env_overrides)
        carla_worker_count = (
            config.env_worker_count - config.nuplan_worker_count
        )
        worker_env_overrides = ()
        if config.nuplan_worker_count > 0:
            worker_env_overrides = (
                (NUPLAN_ENV_OVERRIDES,) * config.nuplan_worker_count
                + ((),) * carla_worker_count
            )
        env = VecDrive(
            settings["agent_count"],
            config.env_worker_count,
            config.seed,
            env_overrides,
            worker_env_overrides,
        )
    else:
        env, args = make_env(
            "continuous", config.seed, settings["agent_count"], env_overrides
        )
    teacher = Teacher(settings["teacher_checkpoint"], args, config.device)
    student = build_student(
        settings["student"], teacher, settings["student_strings"]
    ).to(config.device)
    obs_dim = env.single_observation_space.shape[0]
    buffer = LabelBuffer(config.buffer_size, obs_dim, config.device,
                         config.seed)
    evaluator = None
    if config.eval_interval_labels > 0:
        evaluator = DistillEvaluator(
            args, num_scenarios=settings["eval_num_scenarios"]
        )
    print(f"output: {config.output_dir}")
    distill(env, teacher, student, buffer, config, evaluator)
    env.close()


if __name__ == "__main__":
    main()
