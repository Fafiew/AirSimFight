from sim.environment import AirCombatEnv
from sim.utils import load_config


def test_env_step_shapes():
    cfg = load_config("config/default.yaml")
    env = AirCombatEnv(cfg, headless=True)
    obs = env.reset()
    for _ in range(3):
        actions = {aid: {"throttle": 0.5, "pitch": 0.0, "yaw": 0.0, "roll": 0.0} for aid in obs if aid.startswith("attacker")}
        obs, rewards, dones, infos = env.step(actions)
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(dones, dict)
        assert isinstance(infos, dict)
