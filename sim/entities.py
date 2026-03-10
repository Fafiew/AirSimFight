import numpy as np


class Entity:
    def __init__(self, id: str, position: np.ndarray):
        self.id = id
        self.position = np.array(position, dtype=float)
        self.orientation = np.array([1.0, 0.0, 0.0], dtype=float)


class Attacker(Entity):
    def __init__(self, id: str, position: np.ndarray, cfg: dict):
        super().__init__(id, position)
        dyn = cfg["attacker_dynamics"]
        self.velocity = np.zeros(3, dtype=float)
        self.mass_empty = float(dyn["mass_empty"])
        self.fuel_capacity = float(dyn["fuel_capacity"])
        self.fuel = float(dyn["fuel_mass_initial"])
        self.engine_thrust_max = float(dyn["engine"]["thrust_max"])
        self.fuel_consumption_rate = float(dyn["engine"]["fuel_consumption_rate"])
        self.drag_coefficient = float(dyn["aerodynamics"]["drag_coefficient"])
        self.reference_area = float(dyn["aerodynamics"]["reference_area"])
        self.can_target_defenders = True
        self.is_pretrained = False
        self.body_id = None
        self.health = 100.0

    def step(self, action: dict, dt: float, physics_world) -> None:
        throttle = float(np.clip(action.get("throttle", 0.0), 0.0, 1.0))
        thrust = self.engine_thrust_max * throttle if self.fuel > 0 else 0.0
        fuel_used = thrust * self.fuel_consumption_rate * dt
        self.fuel = max(0.0, self.fuel - fuel_used)
        mass = self.mass_empty + self.fuel
        if self.body_id is not None:
            physics_world.set_mass(self.body_id, mass)
            v = physics_world.get_velocity(self.body_id)
            drag = -self.drag_coefficient * np.linalg.norm(v) * v
            force = np.array([thrust, 0.0, 0.0]) + drag
            physics_world.apply_force(self.body_id, force.tolist(), self.position.tolist())
            self.velocity = physics_world.get_velocity(self.body_id)

    def apply_damage(self, amount: float) -> None:
        self.health -= float(amount)

    def choose_action(self, observation: dict) -> np.ndarray:
        return np.array([0.5, 0.0, 0.0, 0.0], dtype=float)


class Defender(Entity):
    def __init__(self, id: str, position: np.ndarray, cfg: dict):
        super().__init__(id, position)
        d = cfg["defender_defaults"]
        self.radar_range = float(d["radar_range"])
        self.health = float(d["health"])
        self.launch_cooldown = float(d["cooldown"])
        self.last_launch_time = -1e9
        self.body_id = None
        self.destroyed = False

    def scan(self, environment):
        return [{"id": a.id, "bearing": 0.0, "range": float(np.linalg.norm(a.position - self.position)), "rcs": 1.0, "confidence": 0.5} for a in environment.attackers]

    def launch_interceptor(self, target_id: str, physics_world):
        intr = Interceptor(f"int-{self.id}-{target_id}", self.position.copy(), 600.0, "proportional_navigation", target_id)
        intr.body_id = physics_world.add_entity(intr)
        return intr

    def apply_damage(self, amount: float) -> None:
        self.health -= float(amount)
        if self.health <= 0:
            self.destroyed = True


class Interceptor(Entity):
    def __init__(self, id: str, position: np.ndarray, speed: float, guidance_mode: str, target_id: str):
        super().__init__(id, position)
        self.speed = speed
        self.guidance_mode = guidance_mode
        self.target_id = target_id
        self.body_id = None

    def update(self, dt: float, physics_world) -> None:
        if self.body_id is not None:
            physics_world.apply_force(self.body_id, [self.speed, 0, 0], self.position.tolist())
