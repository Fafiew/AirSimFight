import numpy as np

from sim.entities import Attacker
from sim.physics import PhysicsWorld
from sim.utils import load_config


def test_force_increases_velocity_and_mass_updates():
    cfg = load_config("config/default.yaml")
    w = PhysicsWorld(cfg, headless=True)
    a = Attacker("a", np.array([0.0, 0.0, 1000.0]), cfg)
    a.body_id = w.add_entity(a)
    m0 = a.mass_empty + a.fuel
    for _ in range(30):
        a.step({"throttle": 1.0, "attitude": {}}, cfg["sim"]["timestep"], w)
        w.step(cfg["sim"]["timestep"])
    v = w.get_velocity(a.body_id)
    assert np.linalg.norm(v) > 0.01
    m1 = a.mass_empty + a.fuel
    assert m1 < m0
