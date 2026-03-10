import numpy as np

from sim.entities import Attacker
from sim.physics import PhysicsWorld
from sim.utils import load_config


def test_fuel_consumption_and_zero_fuel_thrust():
    cfg = load_config("config/default.yaml")
    w = PhysicsWorld(cfg, headless=True)
    a = Attacker("a", np.array([0.0, 0.0, 1000.0]), cfg)
    a.fuel = 1.0
    a.body_id = w.add_entity(a)
    dt = cfg["sim"]["timestep"]
    f0 = a.fuel
    a.step({"throttle": 1.0, "attitude": {}}, dt, w)
    expected = a.engine_thrust_max * a.fuel_consumption_rate * dt
    actual = f0 - a.fuel
    assert abs(actual - expected) / expected < 0.05

    a.fuel = 0.0
    v0 = w.get_velocity(a.body_id).copy()
    a.step({"throttle": 1.0, "attitude": {}}, dt, w)
    w.step(dt)
    v1 = w.get_velocity(a.body_id)
    assert np.linalg.norm(v1 - v0) < 5.0
    assert not hasattr(a, "max_speed")
