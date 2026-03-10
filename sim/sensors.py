import numpy as np


class Radar:
    def __init__(self, range: float, h_fov_deg: float, v_fov_deg: float, rcs_noise_std: float):
        self.range = range
        self.h_fov = np.deg2rad(h_fov_deg)
        self.v_fov = np.deg2rad(v_fov_deg)
        self.rcs_noise_std = rcs_noise_std

    def scan(self, owner_entity, physics_world, all_entities):
        det = []
        fwd = owner_entity.orientation / (np.linalg.norm(owner_entity.orientation) + 1e-9)
        for ent in all_entities:
            if ent.id == owner_entity.id:
                continue
            rel = ent.position - owner_entity.position
            dist = np.linalg.norm(rel)
            if dist > self.range or dist == 0:
                continue
            bearing = np.arccos(np.clip(np.dot(rel / dist, fwd), -1.0, 1.0))
            if bearing > self.h_fov / 2:
                continue
            hit = physics_world.raycast(owner_entity.position.tolist(), rel.tolist(), dist)
            if hit and hit["body_id"] != ent.body_id:
                continue
            rcs = float(max(0.0, 1.0 + np.random.normal(0, self.rcs_noise_std)))
            confidence = float(np.clip(1.0 - dist / self.range, 0.0, 1.0))
            det.append({"id": ent.id, "distance": float(dist), "bearing": float(bearing), "rcs": rcs, "confidence": confidence})
        return det


class InfraredSeeker:
    def __init__(self, range: float, lock_angle_deg: float, noise_std: float, ir_signature: float):
        self.range = range
        self.lock_angle = np.deg2rad(lock_angle_deg)
        self.noise_std = noise_std
        self.ir_signature = ir_signature

    def get_signal(self, owner_entity, physics_world, all_entities):
        fwd = owner_entity.orientation / (np.linalg.norm(owner_entity.orientation) + 1e-9)
        best = None
        for ent in all_entities:
            if ent.id == owner_entity.id:
                continue
            rel = ent.position - owner_entity.position
            dist = np.linalg.norm(rel)
            if dist > self.range or dist == 0:
                continue
            ang = np.arccos(np.clip(np.dot(rel / dist, fwd), -1.0, 1.0))
            if ang > self.lock_angle:
                continue
            c = max(0.0, (1.0 - ang / self.lock_angle) * (1.0 - dist / self.range)) + np.random.normal(0, self.noise_std)
            if c > 0.2 and (best is None or c > best["confidence"]):
                best = {"id": ent.id, "distance": float(dist), "confidence": float(np.clip(c, 0.0, 1.0))}
        return best or {}
