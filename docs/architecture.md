# AirSimFight Architecture

## Overview

AirSimFight is a 3D reinforcement learning air combat simulation that implements a complete training pipeline from attacker pretraining to dual-agent combat scenarios.

## System Components

### 1. Simulation Layer (`sim/`)

#### Physics Engine (`sim/physics.py`)
- Wraps PyBullet for rigid body physics
- Handles collision detection, force application, and state updates
- Supports both DIRECT (headless) and GUI modes
- Implements substepping for numerical stability

#### Entities (`sim/entities.py`)
- **Entity**: Base class for all simulation objects
- **Attacker**: Aircraft with thrust/fuel dynamics, no artificial max speed
- **Defender**: Ground-based interceptor launcher with radar
- **Interceptor**: Missile/projectile with guidance logic

#### Sensors (`sim/sensors.py`)
- **Radar**: Cone-based detection with RCS modeling and raycast occlusion
- **InfraredSeeker**: Heat-seeking sensor with lock angle and noise

#### Environment (`sim/environment.py`)
- Gym-style environment compatible with Stable-Baselines3
- Multi-agent support with dict-based observations/actions
- Procedural episode generation for dual training

#### JSON Loader (`sim/json_loader.py`)
- Validates visualization JSONs (positions only)
- Rejects entity files with dynamics data
- Schema validation using jsonschema

### 2. Reinforcement Learning (`rl/`)

#### Policies (`rl/policies.py`)
- Custom SB3-compatible policy networks
- Continuous action space for attacker control
- Defender action spaces for scanning and launching

#### Training Scripts
- **trainer_pretrain.py**: Single-agent attacker pretraining
- **trainer_dual.py**: Multi-agent dual training with procedural spawns

#### Model I/O (`rl/model_io.py`)
- Atomic save/load for model checkpoints
- Support for both SB3 (.zip) and PyTorch (.pth) formats

#### RLlib Adapter (`rl/rllib_adapter.py`)
- Optional Ray RLlib integration for scaling
- Not required for basic usage

### 3. Visualization (`viz/`)

#### Renderer (`viz/renderer.py`)
- Panda3D-based 3D rendering
- Displays entities, sensor cones, HUD elements
- Keyboard controls for pause/reset/export

#### Recorder (`viz/recorder.py`)
- Frame capture to PNG sequence
- FFmpeg encoding to MP4

### 4. Tools (`tools/`)

- **evaluate.py**: Model evaluation script
- **visualize.py**: Standalone visualization tool

## Reward Shaping

### Attacker Rewards
- `+target_reward` (100.0): Reaching main target
- `-destroyed_penalty` (-100.0): Attacker destroyed
- `-fuel_cost_coeff * fuel_used`: Fuel efficiency penalty
- Distance-based shaping: Small reward for reducing distance to target

### Defender Rewards
- Positive: Destroying attackers (scaled by value)
- Negative: Defenders destroyed

## Training Phases

### Phase A: Attacker Pretraining
- Single attacker, no defenders
- Random start positions in training corridor
- Objective: Reach target while managing fuel
- Output: Pretrained attacker model

### Phase B: Dual-Agent Training
- Procedural spawn: n_attackers ∈ [1,10], n_defenders ∈ [1,10]
- Defenders in inner ring near target
- Attackers in outer ring at altitude
- Load pretrained attacker policy
- Optional: Freeze attacker weights during defender training

## Configuration

All parameters configurable via `config/default.yaml`:
- Physics: timestep, gravity, drag
- Dynamics: mass, fuel capacity, thrust, consumption rate
- Sensors: range, FOV angles, noise parameters
- RL: algorithm, learning rate, gamma, timesteps
- Device: cpu/cuda selection

CLI overrides via `--overrides '{"key": "value"}'`

## Device Selection

- Explicit `--device` flag (cpu/cuda)
- CUDA fallback to CPU with warning if unavailable
- All PyTorch code uses explicit device mapping

## Safety

- `--safety-check` flag requires acknowledgment
- `--force` flag to proceed after safety check
- Models are simulation-only, not deployable to real hardware
