# Behaviour Cloning - Copy Your Robot

This folder contains 3 simple scripts to teach a new robot by copying an expert.

## What is Behaviour Cloning?

Instead of training a robot from scratch (which takes hours), you:
1. Record an expert robot walking
2. Train a new robot to copy those actions
3. Done! The new robot can walk too

## Quick Start

Run these commands from the **project root** (`/home/mira/isaaclab_ws/biped`):

### Step 1: Record Expert Data
```bash
cd /home/mira/isaaclab_ws/biped
python behaviour_cloning/collect_data.py --load_run YOUR_RUN_NAME --steps 500
```
This creates: `behaviour_cloning/expert_data.pt`

### Step 2: Train Copycat Robot
```bash
python behaviour_cloning/train_bc.py --epochs 100
```
This creates: `behaviour_cloning/bc_policy.pt`

### Step 3: Test Copycat Robot
```bash
python behaviour_cloning/play_bc.py --steps 1000
```
Watch if it walks like the expert!

## Files

| File | What it does |
|------|--------------|
| `collect_data.py` | Records expert robot walking |
| `train_bc.py` | Trains copycat from recorded data |
| `play_bc.py` | Tests the copycat robot |
| `expert_data.pt` | Saved expert demonstrations (auto-created) |
| `bc_policy.pt` | Trained copycat robot (auto-created) |

## Finding Your Run Name

```bash
ls logs/rsl_rl/bipedal_locomotion/
```

Use the folder name you see as `YOUR_RUN_NAME`.

## Example

```bash
# 1. Collect data
python behaviour_cloning/collect_data.py --load_run 2025-03-27_12-34-56 --steps 500

# 2. Train copycat  
python behaviour_cloning/train_bc.py

# 3. Test copycat
python behaviour_cloning/play_bc.py
```
