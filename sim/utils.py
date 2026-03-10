import json
from copy import deepcopy
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def deep_merge(base, override):
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _default_config():
    return {
        "sim": {"timestep": 0.02, "timestep_substep": 0.005, "world_size": 20000, "boundary_timeout_seconds": 5},
        "physics": {"gravity": -9.81, "drag_coefficient_default": 0.01},
        "attacker_dynamics": {"mass_empty": 120.0, "fuel_capacity": 50.0, "fuel_mass_initial": 50.0, "engine": {"thrust_max": 8000.0, "throttle_response_time": 0.1, "fuel_consumption_rate": 0.00012}, "aerodynamics": {"drag_coefficient": 0.02, "reference_area": 0.8, "air_density": 1.225}, "structural": {"enable_stress": True, "max_dynamic_pressure": 5e5}, "fuel_warning_threshold": 0.15},
        "defender_defaults": {"radar_range": 15000.0, "interceptor_speed": 600.0, "cooldown": 2.0, "health": 100.0, "radar": {"horizontal_fov_deg": 120.0, "vertical_fov_deg": 30.0}},
        "procedural_spawn": {"R_def_min": 50.0, "R_def_max": 500.0, "R_att_min": 5000.0, "R_att_max": 20000.0, "H_att_min": 1500.0, "H_att_max": 5000.0},
        "sensors": {"radar": {"default_range": 15000.0, "horizontal_fov_deg": 120.0, "vertical_fov_deg": 30.0, "rcs_noise_std": 0.1}, "ir": {"default_range": 5000.0, "lock_angle_deg": 10.0, "noise_std": 0.05}},
        "rl": {"algorithm": "PPO", "learning_rate": 3e-4, "gamma": 0.99, "ent_coef": 0.01, "n_steps": 2048, "batch_size": 64, "total_timesteps_pretrain": 1000000, "total_timesteps_dual": 2000000, "attacker_rewards": {"target_reward": 100.0, "destroyed_penalty": -100.0, "fuel_cost_coeff": 50.0}},
        "device": "cuda",
        "logging": {"tensorboard": True, "log_dir": "logs/"},
    }


def load_config(path: str, overrides_json: str | None = None) -> dict:
    if yaml is None:
        cfg = _default_config()
    else:
        cfg = yaml.safe_load(Path(path).read_text())
    if overrides_json:
        cfg = deep_merge(cfg, json.loads(overrides_json))
    return cfg
