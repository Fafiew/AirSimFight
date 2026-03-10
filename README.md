# AirSimFight - 3D Reinforcement Learning Air Combat Simulation

A complete air combat simulation with attacker pretraining and dual-agent training, featuring realistic thrust/fuel flight dynamics, computed sensor FOVs, and Panda3D visualization.

## Features

- **Two-phase training**: Attacker pretraining → Dual-agent combat training
- **Procedural episodes**: 1-10 attackers and 1-10 defenders per episode
- **Thrust + Fuel dynamics**: No artificial max speed; physics-driven by engine thrust and fuel consumption
- **Computed sensor FOVs**: Radar and IR sensors with geometric cone detection and raycast occlusion
- **Panda3D visualization**: Real-time 3D rendering with MP4 export via FFmpeg
- **CPU/CUDA modes**: Explicit device selection with automatic fallback
- **SB3 primary RL**: Stable-Baselines3 for PPO/SAC training
- **Optional Ray RLlib**: Adapter for multi-agent scaling

## Quickstart

### Installation

```bash
# Install system dependencies (FFmpeg, CUDA toolkit if needed)
# On Ubuntu:
sudo apt-get install ffmpeg

# Install Python dependencies
pip install -r requirements.txt
```

### Training

**Pretrain attacker (CPU):**
```bash
python -m rl.trainer_pretrain --config config/default.yaml --device cpu --timesteps 1000000 --output models/attacker_pretrained --safety-check --force
```

**Pretrain attacker (GPU):**
```bash
python -m rl.trainer_pretrain --config config/default.yaml --device cuda --timesteps 5000000 --output models/attacker_pretrained
```

**Dual training (load pretrained):**
```bash
python -m rl.trainer_dual --config config/default.yaml --device cuda --timesteps 2000000 --attacker-model models/attacker_pretrained.zip --fine_tune_attacker
```

### Visualization

```bash
python tools/visualize.py --scenario scenarios/example_visualization.json --export out.mp4
```

### TensorBoard

```bash
tensorboard --logdir logs/
```

## System Dependencies

- **FFmpeg**: For MP4 video export (`sudo apt-get install ffmpeg`)
- **CUDA** (optional): For GPU acceleration (`conda install cudatoolkit` or install via system package manager)
- **Panda3D**: For 3D visualization (`pip install panda3d`)

## JSON Visualization Format

Visualization JSONs contain only position data for entities:

```json
{
  "world": {"size": 20000},
  "attackers": [{"id": "a1", "position": [1000, 0, 2000]}],
  "defenders": [{"id": "d1", "position": [0, 0, 0]}],
  "targets": [{"id": "t1", "position": [0, 0, 0]}]
}
```

## Procedural Training Rules

- **Defenders**: Spawned in inner ring (R_def_min to R_def_max) around target on ground
- **Attackers**: Spawned in outer ring (R_att_min to R_att_max) with altitudes between H_att_min and H_att_max
- **Episode sampling**: n_attackers and n_defenders sampled uniformly from [1, 10] each episode

## Attacker Dynamics & Fuel

Attackers use realistic thrust + fuel physics:
- **mass_empty**: Base aircraft mass (kg)
- **fuel_capacity**: Maximum fuel load (kg)
- **engine_thrust_max**: Maximum engine thrust (N)
- **fuel_consumption_rate**: Fuel burn per Newton-second of thrust

When fuel reaches zero, engine produces zero thrust. There is **no artificial max_speed** - speed is limited only by thrust, drag, and gravity.

## FOV & Sensors

### Radar
- Geometric cone detection based on horizontal/vertical FOV
- Raycast occlusion checking
- RCS-based detection probability with noise

### Infrared Seeker
- Narrow lock angle for heat-seeking
- Angle-weighted signal strength
- Noise injection for realism

## Visualization

The Panda3D renderer displays:
- Attackers: Cones with forward axis, fuel bar, throttle HUD
- Defenders: Towers with health bars
- Targets: Ground markers
- Interceptors: Fast projectiles with trails
- Sensor cones: Translucent visualization

Controls:
- `p` - pause/unpause
- `r` - reset scenario
- `v` - toggle sensor visualization
- `e` - export to MP4

## Safety & Ethics

⚠️ **IMPORTANT**: This project is strictly a **simulation and research tool**. 

- Do NOT use to develop real-world weapons or munitions
- Do NOT transfer models to real hardware
- Run on isolated compute when possible
- See `safe_use.md` for full safety guidelines

## Troubleshooting

### "Panda3D not found"
```bash
pip install panda3d
```

### "FFmpeg not found"
```bash
sudo apt-get install ffmpeg  # Ubuntu
brew install ffmpeg          # macOS
```

### CUDA unavailable
The code automatically falls back to CPU if CUDA is unavailable. Use `--device cpu` to force CPU mode.

## Project Structure

```
AirSimFight/
├── config/          # Configuration YAML files
├── scenarios/      # Visualization JSON scenarios
├── sim/            # Physics, entities, sensors, environment
├── rl/             # RL policies and trainers
├── viz/            # Panda3D renderer and recorder
├── tools/          # Evaluation and visualization scripts
├── tests/          # Unit tests
├── examples/       # Example run scripts
└── models/         # Saved model checkpoints
```
