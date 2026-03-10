import random

import numpy as np

try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover
    class _Env: pass
    class _Box:
        def __init__(self, low, high, shape=None, dtype=None):
            self.low=low; self.high=high; self.shape=shape; self.dtype=dtype
    class _Spaces: Box=_Box
    class _Gym: Env=_Env
    gym=_Gym()
    spaces=_Spaces()

from sim.entities import Attacker, Defender, Entity
from sim.json_loader import load_positions
from sim.physics import PhysicsWorld
from sim.sensors import InfraredSeeker, Radar


class AirCombatEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: dict, scenario_json: str = None, device: str = "cpu", headless: bool = True):
        super().__init__()
        self.config = config
        self.device = device
        self.headless = headless
        self.scenario_json = scenario_json
        self.physics_world = PhysicsWorld(config, headless=headless)
        self.attackers = []
        self.defenders = []
        self.interceptors = []
        self.targets = [Entity("target-0", np.array([0.0, 0.0, 0.0]))]
        self.radar = Radar(config["sensors"]["radar"]["default_range"], config["sensors"]["radar"]["horizontal_fov_deg"], config["sensors"]["radar"]["vertical_fov_deg"], config["sensors"]["radar"]["rcs_noise_std"])
        self.ir = InfraredSeeker(config["sensors"]["ir"]["default_range"], config["sensors"]["ir"]["lock_angle_deg"], config["sensors"]["ir"]["noise_std"], 1.0)
        self.action_space = spaces.Box(low=np.array([0, -1, -1, -1], dtype=np.float32), high=np.array([1, 1, 1, 1], dtype=np.float32))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self.t = 0.0

    def _spawn_procedural(self, n_attackers=1, n_defenders=1):
        self.attackers = []
        self.defenders = []
        ps = self.config["procedural_spawn"]
        for i in range(n_attackers):
            r = random.uniform(ps["R_att_min"], ps["R_att_max"])
            theta = random.uniform(0, 2 * np.pi)
            h = random.uniform(ps["H_att_min"], ps["H_att_max"])
            pos = np.array([r * np.cos(theta), r * np.sin(theta), h])
            a = Attacker(f"attacker-{i}", pos, self.config)
            a.body_id = self.physics_world.add_entity(a)
            self.attackers.append(a)
        for i in range(n_defenders):
            r = random.uniform(ps["R_def_min"], ps["R_def_max"])
            theta = random.uniform(0, 2 * np.pi)
            pos = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
            d = Defender(f"defender-{i}", pos, self.config)
            d.body_id = self.physics_world.add_entity(d)
            self.defenders.append(d)

    def reset(self):
        if self.scenario_json and not self.headless:
            data = load_positions(self.scenario_json)
            self.attackers = [Attacker(e["id"], np.array(e["position"]), self.config) for e in data["attackers"]]
            self.defenders = [Defender(e["id"], np.array(e["position"]), self.config) for e in data["defenders"]]
            for e in self.attackers + self.defenders:
                e.body_id = self.physics_world.add_entity(e)
        else:
            self._spawn_procedural(1, 1)
        self.t = 0.0
        return self._build_obs()

    def _build_obs(self):
        obs = {}
        all_entities = self.attackers + self.defenders + self.targets
        for a in self.attackers:
            radar = self.radar.scan(a, self.physics_world, all_entities)
            ir = self.ir.get_signal(a, self.physics_world, all_entities)
            obs[a.id] = {"position": a.position.tolist(), "velocity": a.velocity.tolist(), "fuel": a.fuel, "mass": a.mass_empty + a.fuel, "radar": radar, "ir": ir}
        for d in self.defenders:
            obs[d.id] = {"position": d.position.tolist(), "hp": d.health, "radar": d.scan(self)}
        return obs

    def step(self, actions: dict):
        dt = self.config["sim"]["timestep"]
        rewards, dones, infos = {}, {}, {}
        fuel_coeff = self.config["rl"]["attacker_rewards"]["fuel_cost_coeff"]
        for a in self.attackers:
            ac = actions.get(a.id, {"throttle": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0})
            f0 = a.fuel
            a.step({"throttle": ac.get("throttle", 0.0), "attitude": {"pitch": ac.get("pitch", 0.0), "yaw": ac.get("yaw", 0.0), "roll": ac.get("roll", 0.0)}}, dt, self.physics_world)
            dist = float(np.linalg.norm(a.position - self.targets[0].position))
            rewards[a.id] = -fuel_coeff * max(0.0, f0 - a.fuel) - 0.0001 * dist
            dones[a.id] = False
            infos[a.id] = {}
        for d in self.defenders:
            rewards[d.id] = 0.0
            dones[d.id] = d.health <= 0
            infos[d.id] = {}
        self.physics_world.step(dt)
        self.t += dt
        dones["__all__"] = self.t > 2.0
        return self._build_obs(), rewards, dones, infos

    def render(self, mode="human"):
        if mode == "human" and not self.headless:
            try:
                from viz.renderer import PandaRenderer

                PandaRenderer(self.config).draw(self)
            except Exception:
                return None

    def close(self):
        return None
