"""scripts/distill/train.py runs end to end from the command line."""

import json
import os
import subprocess
import sys

import pytest

from pufferlib.distill.env import REPO_ROOT

from conftest import needs_teacher

SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "distill", "train.py")
SCRIPT_TIMEOUT_SECONDS = 300
ITERATIONS = 2


@pytest.mark.slow
@needs_teacher
def test_train_script_end_to_end(
        tmp_path,
    ) -> None:
    """Two BC iterations write a checkpoint and a log with labels_seen."""
    arguments = [
        f"iterations={ITERATIONS}",
        "agent_count=64",
        "env_steps_per_iteration=10",
        "gradient_steps_per_iteration=2",
        "batch_size=64",
        "buffer_size=10000",
        "eval_interval_labels=0",
        f"output_dir={tmp_path}",
    ]
    completed = subprocess.run(
        [sys.executable, SCRIPT_PATH, *arguments],
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_SECONDS,
        cwd=tmp_path,  # the script must not depend on the cwd
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert (tmp_path / "student.pt").is_file()
    rows = [
        json.loads(line)
        for line in (tmp_path / "log.jsonl").read_text().splitlines()
    ]
    assert len(rows) == ITERATIONS
    assert rows[-1]["labels_seen"] > rows[0]["labels_seen"] > 0


def test_train_script_rejects_unknown_keys(
        tmp_path,
    ) -> None:
    """A typo in a key fails instead of being ignored."""
    completed = subprocess.run(
        [sys.executable, SCRIPT_PATH, "iteratons=2"],
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_SECONDS,
        cwd=tmp_path,
    )
    assert completed.returncode != 0
    assert "unknown key 'iteratons'" in completed.stderr


@pytest.mark.slow
@needs_teacher
def test_train_script_mpc_smoke(
        tmp_path,
    ) -> None:
    """student=mpc with mpc.* settings runs and logs wrong-side
    saturation."""
    arguments = [
        "student=mpc",
        "mpc.horizon_steps=3",
        "mpc.act_warm_start=false",
        "iterations=2",
        "agent_count=64",
        "env_steps_per_iteration=10",
        "gradient_steps_per_iteration=2",
        "batch_size=64",
        "buffer_size=10000",
        "eval_interval_labels=0",
        f"output_dir={tmp_path}",
    ]
    completed = subprocess.run(
        [sys.executable, SCRIPT_PATH, *arguments],
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_SECONDS,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    rows = [
        json.loads(line)
        for line in (tmp_path / "log.jsonl").read_text().splitlines()
    ]
    assert len(rows) == ITERATIONS
    assert "wrong_side_saturation" in rows[-1]


def test_train_script_rejects_settings_for_another_student(
        tmp_path,
    ) -> None:
    """mpc.* settings with the MLP student fail instead of being ignored."""
    completed = subprocess.run(
        [sys.executable, SCRIPT_PATH, "student=mlp", "mpc.horizon_steps=5"],
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_SECONDS,
        cwd=tmp_path,
    )
    assert completed.returncode != 0
    assert "needs student=mpc" in completed.stderr
