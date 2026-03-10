import argparse
import random
from pathlib import Path


from sim.environment import AirCombatEnv
from sim.utils import load_config


def safety_gate(force: bool):
    print(Path("safe_use.md").read_text())
    if not force:
        raise SystemExit("Refusing to run without --force when --safety-check is enabled.")


def pick_device(req: str):
    try:
        import torch
    except Exception:
        if req == "cuda":
            print("WARNING: CUDA requested but PyTorch unavailable; falling back to CPU.")
        return "cpu"
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
    ap.add_argument("--attacker-model", required=True)
    ap.add_argument("--fine_tune_attacker", action="store_true")
    ap.add_argument("--safety-check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--overrides", default=None)
    args = ap.parse_args()
    if args.safety_check:
        safety_gate(args.force)
    cfg = load_config(args.config, args.overrides)
    _device = pick_device(args.device)
    total = args.timesteps or int(cfg["rl"]["total_timesteps_dual"])
    env = AirCombatEnv(cfg, device=str(_device), headless=True)

    steps = 0
    while steps < total:
        n_att = random.randint(1, 10)
        n_def = random.randint(1, 10)
        env._spawn_procedural(n_att, n_def)
        obs = env._build_obs()
        for _ in range(100):
            actions = {aid: {"throttle": random.random(), "pitch": 0.0, "yaw": 0.0, "roll": 0.0} for aid in [a.id for a in env.attackers]}
            obs, _, dones, _ = env.step(actions)
            steps += 1
            if dones.get("__all__") or steps >= total:
                break
    print(f"Dual training loop complete. fine_tune_attacker={args.fine_tune_attacker}, attacker_model={args.attacker_model}")


if __name__ == "__main__":
    main()
