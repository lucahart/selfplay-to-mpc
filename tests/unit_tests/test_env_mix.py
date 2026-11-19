"""env_mix: the first env_mix.num_envs envs take extra env overrides."""

import pytest

import pufferlib.vector
from pufferlib.pufferl import env_kwargs_per_env, load_env


def make_args(mixed_count, mixed_env):
    return {
        "env": {"simulation_mode": "replay", "map_dir": "data/nuplan"},
        "vec": {"num_envs": 4},
        "env_mix": {"num_envs": mixed_count, "env": mixed_env},
    }


def test_no_mix_gives_every_env_the_same_kwargs():
    kwargs = env_kwargs_per_env(make_args(0, {}))
    assert kwargs == [{"simulation_mode": "replay", "map_dir": "data/nuplan"}] * 4


def test_missing_section_gives_every_env_the_same_kwargs():
    args = make_args(0, {})
    del args["env_mix"]
    assert len(env_kwargs_per_env(args)) == 4


def test_first_envs_take_the_mix_overrides():
    kwargs = env_kwargs_per_env(make_args(1, {"simulation_mode": "gigaflow"}))
    assert kwargs[0] == {"simulation_mode": "gigaflow", "map_dir": "data/nuplan"}
    assert all(k["simulation_mode"] == "replay" for k in kwargs[1:])


@pytest.mark.parametrize("mixed_count", [-1, 4, 5])
def test_mix_must_leave_some_envs_unmixed(mixed_count):
    with pytest.raises(ValueError):
        env_kwargs_per_env(make_args(mixed_count, {"simulation_mode": "gigaflow"}))


def test_mix_needs_overrides():
    with pytest.raises(ValueError):
        env_kwargs_per_env(make_args(2, {}))


@pytest.mark.parametrize("backend", ["PufferEnv", "Multiprocessing"])
def test_load_env_passes_kwargs_in_the_backend_form(monkeypatch, backend):
    # The native backend (puffer eval's probe env) takes one kwargs dict;
    # a per-env list there crashed `puffer eval` after env_mix was added
    calls = []
    monkeypatch.setattr(
        pufferlib.vector,
        "make",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    args = make_args(0, {})
    args["package"] = "ocean"
    num_envs = 1 if backend == "PufferEnv" else 4
    args["vec"] = {"backend": backend, "num_envs": num_envs}
    load_env("puffer_drive", args)
    (creators,), kwargs = calls[0]
    if backend == "PufferEnv":
        assert callable(creators)
        assert kwargs["env_kwargs"] == args["env"]
    else:
        assert len(creators) == 4
        assert kwargs["env_kwargs"] == [args["env"]] * 4
