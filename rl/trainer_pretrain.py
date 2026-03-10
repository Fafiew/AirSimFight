import argparse
from pathlib import Path


def safety_gate(force: bool):
    text = Path("safe_use.md").read_text()
    print(text)
    if not force:
        raise SystemExit("Refusing to run without --force when --safety-check is enabled.")


def pick_device(req: str):
    import torch

    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req == "cuda":
        print("WARNING: CUDA requested but unavailable; falling back to CPU.")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--output", default="models/attacker_pretrained")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--safety-check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--overrides", default=None)
    args = ap.parse_args()

    if args.safety_check:
        safety_gate(args.force)

    from rl.model_io import atomic_save_torch
    from sim.environment import AirCombatEnv
    from sim.utils import load_config

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as e:
        raise SystemExit(f"Stable Baselines3 unavailable: {e}")

    cfg = load_config(args.config, args.overrides)
    device = pick_device(args.device)
    timesteps = args.timesteps or int(cfg["rl"]["total_timesteps_pretrain"])

    def _mk():
        env = AirCombatEnv(cfg, scenario_json=args.scenario, device=str(device), headless=True)
        env._spawn_procedural(1, 0)
        return _SingleAttackerWrapper(env)

    vec_env = DummyVecEnv([_mk])
    model = PPO("MlpPolicy", vec_env, learning_rate=float(cfg["rl"]["learning_rate"]), gamma=float(cfg["rl"]["gamma"]), ent_coef=float(cfg["rl"]["ent_coef"]), n_steps=int(cfg["rl"]["n_steps"]), batch_size=int(cfg["rl"]["batch_size"]), tensorboard_log=cfg["logging"]["log_dir"] if cfg["logging"]["tensorboard"] else None, device=str(device), verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(args.output)
    atomic_save_torch(model.policy.state_dict(), f"{args.output}.pth")


class _SingleAttackerWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.agent_id = None

    def reset(self):
        obs = self.env.reset()
        self.agent_id = next(iter(obs.keys()))
        return self._vec(obs[self.agent_id])

    def step(self, action):
        a = {self.agent_id: {"throttle": float(action[0]), "pitch": float(action[1]), "yaw": float(action[2]), "roll": float(action[3])}}
        obs, rew, done, info = self.env.step(a)
        return self._vec(obs[self.agent_id]), rew[self.agent_id], done["__all__"], info[self.agent_id]

    def _vec(self, obs):
        import numpy as np

        arr = np.zeros(10, dtype=np.float32)
        arr[:3] = obs.get("position", [0, 0, 0])
        arr[3:6] = obs.get("velocity", [0, 0, 0])
        arr[6] = obs.get("fuel", 0)
        arr[7] = obs.get("mass", 0)
        arr[8] = len(obs.get("radar", []))
        arr[9] = obs.get("ir", {}).get("confidence", 0) if isinstance(obs.get("ir", {}), dict) else 0
        return arr


if __name__ == "__main__":
    main()
