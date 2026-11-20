# From Self-Play RL to Constrained Control: Distilling DiffMPC Students with DAgger

Luca M. Hartmann and Parinaz Naghizadeh, University of California San Diego

**[Project website](https://lucahart.github.io/selfplay-to-mpc/)** · videos, method overview and results

We add explicit, checkable constraints to a strong end-to-end driving policy
without retraining it from scratch. A teacher is trained with self-play RL in
[PufferDrive](https://github.com/Emerge-Lab/PufferDrive), its action head is
replaced by a constrained differentiable MPC, and the new policy is distilled
from the teacher with DAgger in the same simulator. The constraints are built
from the scene, not learned: road boundary half-planes anchored on the route,
and ellipses around the four nearest agents.

## Results

Failure rate (collisions plus off-road per vehicle, lower is better), mean ±
standard error over nine seeds. Waymo is never used for training.

| Teacher | Policy | CARLA | nuPlan | Waymo |
|---|---|---|---|---|
| Small (4x512) | Teacher | 0.245 | 0.181 | 0.298 |
| | MLP | 0.226 ± 0.004 | 0.205 ± 0.006 | 0.299 ± 0.007 |
| | Unconstrained DiffMPC | 0.224 ± 0.003 | 0.220 ± 0.005 | 0.280 ± 0.008 |
| | Constrained DiffMPC | 0.180 ± 0.003 | 0.180 ± 0.006 | 0.261 ± 0.008 |
| Large (4x1024) | Teacher | 0.052 | 0.039 | 0.059 |
| | MLP | 0.050 ± 0.001 | 0.040 ± 0.001 | 0.059 ± 0.002 |
| | Unconstrained DiffMPC | 0.052 ± 0.002 | 0.044 ± 0.003 | 0.059 ± 0.002 |
| | Constrained DiffMPC | 0.053 ± 0.001 | 0.035 ± 0.001 | 0.046 ± 0.002 |

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
python setup.py build_ext --inplace --force
```

The DiffMPC students solve their MPC with
[WarpMPC](https://github.com/hshose/WarpMPC) on the GPU, which needs JAX,
Warp and CasADi. `docs/setup.md` lists the exact versions, how to get the
nuPlan and Waymo test scenes, and optional packages.

## Quickstart

```bash
# Distill a constrained DiffMPC student from the large teacher (seed 1)
scripts/paper/train_student.sh large constrained 1

# Evaluate its best checkpoint on CARLA, nuPlan and Waymo
scripts/paper/evaluate_student.sh large constrained 1

# Collect the results into the table above
python scripts/paper/make_table.py
```

The teacher checkpoints go in `checkpoints/` (`teacher_4x512.pt` and
`teacher_4x1024.pt`). To train a teacher yourself, run
`puffer train puffer_drive`.

## Reproducing the paper

`scripts/paper/run_all.sh` evaluates both teachers, then trains and
evaluates 2 teachers x 3 students x 9 seeds, and prints the table. The
settings of every run are in `scripts/paper/settings.sh`. On one RTX 5090 a
DiffMPC student takes 2 to 5 hours to distill.

## Repository layout

| Path | Contents |
|---|---|
| `pufferlib/distill/` | Students, MPC problem and solver interface, constraints, DAgger loop |
| `scripts/distill/` | Training, evaluation, teacher check and plotting entry points |
| `scripts/paper/` | The paper's settings and runs |
| `pufferlib/ocean/drive/` | The PufferDrive simulator (C) and its Python wrapper |
| `pufferlib/config/puffer_drive.yaml` | Simulator, teacher training and evaluator settings |
| `tests/distill/` | Tests of the distillation code (`python -m pytest tests/distill -q`) |
| `docs/` | How the code implements the method, setup, evaluation |
| `website/` | Source of the project website |

## Generative AI use

The research questions, method, experimental design, and analysis of this
work are the authors' own. Generative AI tools wrote this codebase for the
experiments, under the authors' specification and review. The authors
verified all AI-generated code and take full responsibility for the content
of this repository and its corresponding paper.

## Citation

```bibtex
@misc{hartmann2026selfplay,
  title         = {From Self-Play {RL} to Constrained Control:
                   Distilling {DiffMPC} Students with {DAgger}},
  author        = {Hartmann, Luca M. and Naghizadeh, Parinaz},
  year          = {2026},
  eprint        = {2611.25197},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2611.25197}
}
```

## Acknowledgments

This repository is a fork of
[PufferDrive](https://github.com/Emerge-Lab/PufferDrive), the self-play
driving simulator built on [PufferLib](https://github.com/PufferAI/PufferLib).
The MPC is solved with [WarpMPC](https://github.com/hshose/WarpMPC). The code
is released under the MIT license (`LICENSE`).
