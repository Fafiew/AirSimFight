import logging
from typing import Sequence

import numpy as np

try:
    import pybullet as p
except Exception:  # pragma: no cover
    p = None


class PhysicsWorld:
    def __init__(self, config: dict, headless: bool = True):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.entity_map = {}
        self._fallback_state = {}
        self._next_id = 1
        self._use_fallback = p is None
        if not self._use_fallback:
            mode = p.DIRECT if headless else p.GUI
            self.client = p.connect(mode)
            p.setGravity(0, 0, float(config["physics"]["gravity"]), physicsClientId=self.client)
        else:
            self.client = None
            self.gravity = np.array([0.0, 0.0, float(config["physics"]["gravity"])])

    def step(self, dt: float) -> None:
        sub = max(1, int(dt / self.config["sim"]["timestep_substep"]))
        if not self._use_fallback:
            for _ in range(sub):
                p.stepSimulation(physicsClientId=self.client)
            return
        sdt = dt / sub
        for _ in range(sub):
            for st in self._fallback_state.values():
                acc = st["force"] / max(st["mass"], 1e-6) + self.gravity
                st["vel"] += acc * sdt
                st["pos"] += st["vel"] * sdt
                st["force"] = np.zeros(3)

    def add_entity(self, entity) -> int:
        if not self._use_fallback:
            shape = p.createCollisionShape(p.GEOM_SPHERE, radius=1.0, physicsClientId=self.client)
            body_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=shape, basePosition=entity.position.tolist(), physicsClientId=self.client)
            self.entity_map[entity.id] = body_id
            return body_id
        body_id = self._next_id
        self._next_id += 1
        self.entity_map[entity.id] = body_id
        self._fallback_state[body_id] = {"pos": np.array(entity.position, dtype=float), "vel": np.zeros(3), "force": np.zeros(3), "mass": 1.0}
        return body_id

    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self.entity_map:
            body_id = self.entity_map[entity_id]
            if not self._use_fallback:
                p.removeBody(body_id, physicsClientId=self.client)
            else:
                self._fallback_state.pop(body_id, None)
            del self.entity_map[entity_id]

    def apply_force(self, entity_body_id: int, force: Sequence[float], pos: Sequence[float]) -> None:
        if not self._use_fallback:
            p.applyExternalForce(entity_body_id, -1, forceObj=force, posObj=pos, flags=p.WORLD_FRAME, physicsClientId=self.client)
            return
        self._fallback_state[entity_body_id]["force"] += np.array(force, dtype=float)

    def get_velocity(self, entity_body_id: int) -> np.ndarray:
        if not self._use_fallback:
            vel, _ = p.getBaseVelocity(entity_body_id, physicsClientId=self.client)
            return np.array(vel, dtype=float)
        return self._fallback_state[entity_body_id]["vel"].copy()

    def set_mass(self, entity_body_id: int, mass: float) -> None:
        if not self._use_fallback:
            p.changeDynamics(entity_body_id, -1, mass=float(mass), physicsClientId=self.client)
            return
        self._fallback_state[entity_body_id]["mass"] = float(mass)

    def raycast(self, origin: Sequence[float], direction: Sequence[float], max_dist: float) -> dict | None:
        d = np.asarray(direction, dtype=float)
        n = np.linalg.norm(d)
        if n == 0:
            return None
        if not self._use_fallback:
            end = (np.asarray(origin) + d / n * max_dist).tolist()
            hit = p.rayTest(origin, end, physicsClientId=self.client)[0]
            if hit[0] < 0:
                return None
            return {"body_id": hit[0], "hit_pos": hit[3], "fraction": hit[2]}
        return None
