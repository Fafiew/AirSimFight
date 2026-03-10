"""
AirSimFight - 3D Reinforcement Learning Air Combat Simulation
"""

__version__ = "0.1.0"

from sim.environment import AirCombatEnv
from sim.entities import Entity, Attacker, Defender, Interceptor
from sim.physics import PhysicsWorld
from sim.sensors import Radar, InfraredSeeker
from sim.json_loader import load_positions

__all__ = [
    "AirCombatEnv",
    "Entity", 
    "Attacker",
    "Defender",
    "Interceptor",
    "PhysicsWorld",
    "Radar",
    "InfraredSeeker",
    "load_positions",
]
