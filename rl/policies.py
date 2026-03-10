import numpy as np


def obs_to_vec(obs: dict) -> np.ndarray:
    pos = np.array(obs.get("position", [0, 0, 0]), dtype=float)
    vel = np.array(obs.get("velocity", [0, 0, 0]), dtype=float)
    fuel = np.array([obs.get("fuel", 0.0)], dtype=float)
    return np.concatenate([pos, vel, fuel])
