# AirSimFight

## Project description
AirSimFight is a research-oriented 3D RL air-combat simulation with attacker pretraining and dual-agent procedural training.

## Quickstart
Install deps and run examples below.

## System dependencies
Requires CUDA (optional), FFmpeg, and graphics stack for Panda3D.

## Installation
```bash
pip install -r requirements.txt
```

## Examples
**Pretrain CPU**
```bash
python -m rl.trainer_pretrain --config config/default.yaml --device cpu --timesteps 1000000 --output models/attacker_pretrained --safety-check --force
```

**Pretrain GPU**
```bash
python -m rl.trainer_pretrain --config config/default.yaml --device cuda --timesteps 5000000 --output models/attacker_pretrained
```

**Dual training (load pretrained)**
```bash
python -m rl.trainer_dual --config config/default.yaml --device cuda --timesteps 2000000 --attacker-model models/attacker_pretrained.zip --fine_tune_attacker
```

**Visualize scenario and export**
```bash
python tools/visualize.py --scenario scenarios/example_visualization.json --export out.mp4
```

**TensorBoard**
```bash
tensorboard --logdir logs/
```

## JSON visualization format
Visualization JSON is minimal: only `id` and `position` for entities.

## Procedural training rules
Dual training samples attacker/defender counts uniformly from 1 to 10 each episode.

## Attacker dynamics & fuel
Attackers use thrust + fuel mass depletion dynamics and there is **no max_speed** cap.

## FOV & sensors
Radar and IR detections are computed from geometric cones and raycasts.

## Visualization & MP4 export
Panda3D is used for visualization; FFmpeg is used for MP4 export.

## Safety & ethics
See [safe_use.md](safe_use.md).

## Troubleshooting
If CUDA/Panda3D/FFmpeg are missing, the scripts print clear fallback warnings.
