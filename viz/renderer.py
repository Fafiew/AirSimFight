class PandaRenderer:
    def __init__(self, config: dict):
        self.config = config
        self.paused = False
        self.show_sensors = True

    def draw(self, env):
        return {"attackers": len(env.attackers), "defenders": len(env.defenders), "sensor_cones": self.show_sensors}
