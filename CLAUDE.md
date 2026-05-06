# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Bipedal locomotion RL on a 6-DOF robot (left/right hip pitch, hip roll, knee), using **Isaac Lab manager-based workflow** + **RSL-RL PPO** for training, with **MuJoCo sim2sim** deployment. See `README.md` for end-user usage and `AGENTS.md` for an exhaustive walkthrough; this file captures only the non-obvious bits.

## Common commands

Scripts run from the project root, with the `biped` extension installed (`pip install -e source/biped`) and Isaac Lab on the Python path. If using Isaac Lab's launcher, replace `python` with `./isaaclab.sh -p` from the Isaac Lab dir.

```bash
# Train / resume
python scripts/rsl_rl/train.py --task=biped_walk_flat [--num_envs N] [--max_iterations N] [--resume] [--load_run NAME]

# Play (auto-exports policy.pt + policy.onnx to logs/.../<run>/exported/)
python scripts/rsl_rl/play.py --task=biped_walk_flat_play --load_run RUN_NAME
python scripts/rsl_rl/keyboard_teleport.py  --task=biped_walk_flat_play --load_run RUN_NAME   # WASD
python scripts/rsl_rl/gamepad_teleport.py   --task=biped_walk_flat_play --load_run RUN_NAME

# Inspect policy I/O (obs terms / actions)
python scripts/rsl_rl/debug_policy.py --task=biped_walk_flat_play --checkpoint path/to/model.pt --obs_data_view

# Sim2sim in MuJoCo (edit `relative_policy_path` and `cmd_vel` inside the script)
python sim2sim_mujoco/mujoco_basic_deploy.py        # uses .pt
python sim2sim_mujoco/mujoco_basic_deploy_onnx.py   # uses .onnx
python sim2sim_mujoco/mujoco_keyb_teleport.py
```

Tasks: `biped_walk_{flat,rough,stairs}` and `..._play` variants. There is no test suite — smoke-test by running `random_agent.py` / `zero_agent.py`, or training for `--max_iterations=10`.

## Lint / format

Pre-commit hooks (`pre-commit install` once, then `pre-commit run --all-files`): black (line length 120, `--unstable`), flake8 (with flake8-simplify, flake8-return), isort (black profile), pyupgrade (`--py310-plus`), codespell, and an **insert-license** hook that auto-prepends `.github/LICENSE_HEADER.txt` to every `.py`/`.yml`. New files without that header will be modified by the hook.

## Architecture (the non-obvious bits)

### How environments get registered

The package is auto-discovered, not imported by hand:

- `source/biped/biped/__init__.py` does `from .tasks import *`.
- `tasks/__init__.py` calls `isaaclab_tasks.utils.import_packages(__name__, _BLACKLIST_PKGS)`, which walks subpackages and imports them — that's what triggers the `gym.register(...)` calls in `tasks/locomotion/robots/__init__.py`.
- `_BLACKLIST_PKGS = ["utils", ".mdp"]` — anything inside `mdp/` (observations, rewards, events, curriculums, commands) is **not** auto-imported as a task module. It's only pulled in via explicit imports from cfg files. Don't put `gym.register` calls under `mdp/`.

To add a new task: define a `*EnvCfg` (and `*EnvCfg_PLAY`) in `tasks/locomotion/robots/biped_env_cfg.py`, then add a `gym.register` block in `tasks/locomotion/robots/__init__.py` pointing at it and at `PointFootPPORunnerCfg` (`agents/rsl_rl_ppo_cfg.py`).

### Config inheritance chain

`BipedEnvCfg` (in `cfg/base_env_cfg.py`) is the base — robot + scene fields are `MISSING`. `BipedBaseEnvCfg` (in `robots/biped_env_cfg.py`) fills them in with `BIPED_CONFIG` and sets initial joint pose. `BipedBlind{Flat,Rough,Stair}EnvCfg` then specialize per terrain. Each `*_PLAY` variant inherits from `BipedBaseEnvCfg_PLAY`, which disables observation noise (`enable_corruption=False`) and removes the `push_robot` and `add_base_mass` events. Override knobs in `__post_init__` after calling `super().__post_init__()`, never at class-attribute level.

### Sim2sim invariants (easy to break)

These must agree between training-time config and `sim2sim_mujoco/sim_config.py`, or the policy will silently produce garbage:

- **Joint order** in `Sim2simCfg.robot_config.joint_names` must match the order the training-time `ActionManager` used. The truth is `logs/rsl_rl/bipedal_locomotion/<run>/params/env.yaml`.
- **Observation history length**: `obs_history_len = 5` (sim2sim) must equal `history_length = 5` in the training observation group, and `flatten_history_dim` must agree on both sides — otherwise the obs vector shape mismatches.
- **Action scale** (`0.25`) and **default joint positions** must match (`scripts/test_joint_order.py` is a useful sanity check).
- The MuJoCo XML (`sim2sim_mujoco/mujoco_xml/SF_biped.xml`) is the source of truth for the deployed robot — there is also a `SF_biped copy.xml` floating around (likely WIP); the active script imports `SF_biped.xml`.

### Policy export

`play.py` exports both TorchScript (`policy.pt`) and ONNX (`policy.onnx`) under `logs/rsl_rl/bipedal_locomotion/<run>/exported/`. The exporter lives at `source/biped/biped/utils/wrappers/rsl_rl/exporter.py`. Sim2sim scripts reference these by `relative_policy_path`, hard-coded inside each script — update before running.

### Robot asset

`source/biped/biped/assets/usd/SF_bipedal_usd/` (gitignored) holds the USD model loaded by `BIPED_CONFIG` in `assets/config/simple_biped_config.py`. If `BIPED_CONFIG` fails to spawn, the USD is missing, not the code.

### Dual sim engines, two configs

- **Isaac Sim / PhysX** for training — gains, masses, friction etc. live in `simple_biped_config.py` and event terms in `cfg/base_env_cfg.py`.
- **MuJoCo** for deployment — gains live in `sim_config.py` (`stiffness_gain=40.0`, `damping_gain=-2.5`) and in the XML.

A change in one place doesn't propagate. When tuning physics, update both — and if changing joint order, **stiffness scheme**, or action scale, also re-export and update the XML.

## Output layout

- `logs/rsl_rl/bipedal_locomotion/<run>/` — checkpoints, `params/env.yaml`, `exported/`. Both `logs/` and `outputs/` are gitignored.
- `behaviour_cloning/` — separate experimental BC pipeline (collect → train → play); see `behaviour_cloning/README.md`. Not part of the main RL flow.
