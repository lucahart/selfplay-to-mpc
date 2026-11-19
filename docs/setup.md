# Setup

Commands assume the repository root with the virtual environment active
(`source .venv/bin/activate`).

## Python environment

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
python setup.py build_ext --inplace --force
```

Rebuild the C extension after every `.c` / `.h` change, and after installing
the EGL headers (rendering is only compiled in when `/usr/include/EGL/egl.h`
exists at build time).

## DiffMPC students: JAX, Warp and WarpMPC

The DiffMPC students solve their MPC with
[WarpMPC](https://github.com/hshose/WarpMPC). The paper's runs used JAX
0.11.2 (CUDA 13), Warp 1.18, CasADi 3.8.1 and torch 2.14.1+cu130 on an RTX
5090:

```bash
uv pip install "jax[cuda13]==0.11.2" warp-lang==1.18.0 casadi==3.8.1
git clone https://github.com/hshose/WarpMPC ../WarpMPC
git -C ../WarpMPC checkout e5ddb20a1fcb06453e97f31936b3f7c5164e7ca1
uv pip install -e ../WarpMPC
```

Run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`; otherwise JAX reserves most
of the GPU memory that PyTorch needs. Without WarpMPC the MLP student still
trains and the WarpMPC tests are skipped.

## Data

| Test set | Where it comes from |
|---|---|
| CARLA | Shipped in `pufferlib/resources/drive/binaries/carla`. |
| nuPlan | `uv pip install awscli`, then `python data_utils/fetch_data.py` fetches `nuplan_mini_val` (about 10 GB, public bucket, no credentials) into `data/`. CC BY-NC-SA 4.0. |
| Waymo | The WOMD validation scenes as JSON from the Hugging Face dataset `EMERGE-lab/GPUDrive`, converted with `python data_utils/json_to_bin.py --input <json dir> --output data/womd_val`. |

## Optional packages

| Package | For |
|---|---|
| `sudo apt install libegl-dev xvfb ffmpeg` | Rendering videos. |
| `sudo apt install libgl-dev libx11-dev libegl-dev` | The C tests (`make -C tests/drive test`). |
| `uv pip install pytest` | The Python tests. |
| `uv pip install nvidia-ml-py` | GPU statistics on the training dashboard. |

## Known issues

| Issue | Why |
|---|---|
| Connect with plain `ssh`, not `ssh -X`. | The renderer starts its own virtual display only when `DISPLAY` is unset. |
| A map that fails to load causes a segfault, not an error. | The env init path returns without reporting the failure. |
