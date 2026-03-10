"""
Sensor implementations for radar and infrared detection.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Radar:
    """
    Radar sensor with cone-based detection and raycast occlusion.
    """
    
    def __init__(
        self,
        range: float = 15000.0,
        h_fov_deg: float = 120.0,
        v_fov_deg: float = 30.0,
        rcs_noise_std: float = 0.1
    ):
        """
        Initialize radar sensor.
        
        Args:
            range: Maximum detection range in meters
            h_fov_deg: Horizontal field of view in degrees
            v_fov_deg: Vertical field of view in degrees
            rcs_noise_std: Standard deviation for RCS noise
        """
        self.range = range
        self.h_fov = np.deg2rad(h_fov_deg)
        self.v_fov = np.deg2rad(v_fov_deg)
        self.rcs_noise_std = rcs_noise_std
        
        logger.debug(f"Radar: range={range}, h_fov={h_fov_deg}°, v_fov={v_fov_deg}°")
    
    def scan(
        self,
        owner_entity,
        physics_world,
        all_entities: List
    ) -> List[Dict]:
        """
        Scan for entities within radar cone.
        
        Args:
            owner_entity: Entity that owns this sensor
            physics_world: PhysicsWorld instance
            all_entities: List of all entities to scan
            
        Returns:
            List of detection dicts {id, distance, bearing, rcs, confidence}
        """
        detections = []
        
        if not owner_entity.is_alive:
            return detections
        
        # Get owner position and orientation
        owner_pos = owner_entity.position
        
        # Get forward direction from orientation
        if hasattr(owner_entity, 'body_id') and owner_entity.body_id is not None:
            forward = physics_world.get_forward_vector(owner_entity.body_id)
        else:
            forward = np.array([1, 0, 0])
        
        # Get up direction
        up = np.array([0, 0, 1])
        
        # Check each entity
        for entity in all_entities:
            if entity.id == owner_entity.id:
                continue
            
            if not hasattr(entity, 'is_alive') or not entity.is_alive:
                continue
            
            # Direction to target
            to_target = entity.position - owner_pos
            distance = np.linalg.norm(to_target)
            
            if distance > self.range or distance < 1.0:
                continue
            
            # Normalize direction
            target_dir = to_target / distance
            
            # Check if within horizontal FOV
            cos_angle = np.dot(forward, target_dir)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            if angle > self.h_fov / 2:
                continue
            
            # Check vertical FOV
            # Project onto vertical plane
            forward_horizontal = forward - np.dot(forward, up) * up
            forward_horizontal = forward_horizontal / (np.linalg.norm(forward_horizontal) + 1e-8)
            
            vertical_angle = np.arccos(np.clip(np.dot(target_dir, up), -1, 1))
            
            if vertical_angle > self.v_fov / 2:
                continue
            
            # Raycast occlusion check
            if physics_world is not None:
                hit = physics_world.raycast(owner_pos.tolist(), target_dir.tolist(), distance)
                if hit is not None and hit.get('entity_id') != entity.id:
                    # Occluded
                    continue
            
            # Compute RCS (simplified - use distance-based model)
            rcs = self._compute_rcs(entity, distance)
            
            # Add noise
            rcs_noisy = rcs + np.random.normal(0, self.rcs_noise_std)
            
            # Compute detection confidence based on RCS and range
            confidence = self._compute_confidence(rcs_noisy, distance)
            
            # Only report detections above threshold
            if confidence > 0.1:
                # Compute bearing angle
                bearing = np.arctan2(target_dir[1], target_dir[0])
                
                detections.append({
                    'id': entity.id,
                    'distance': distance,
                    'bearing': bearing,
                    'rcs': rcs_noisy,
                    'confidence': confidence,
                    'position': entity.position.tolist()
                })
        
        return detections
    
    def _compute_rcs(self, entity, distance: float) -> float:
        """
        Compute radar cross section (simplified model).
        
        Args:
            entity: Target entity
            distance: Range to target
            
        Returns:
            RCS value in m²
        """
        # Base RCS based on entity type
        if hasattr(entity, '__class__'):
            if entity.__class__.__name__ == 'Attacker':
                base_rcs = 1.0
            elif entity.__class__.__name__ == 'Defender':
                base_rcs = 5.0
            elif entity.__class__.__name__ == 'Interceptor':
                base_rcs = 0.1
            else:
                base_rcs = 1.0
        else:
            base_rcs = 1.0
        
        # Distance falloff (simplified)
        distance_falloff = (1000.0 / distance) ** 0.5
        
        return base_rcs * distance_falloff
    
    def _compute_confidence(self, rcs: float, distance: float) -> float:
        """
        Compute detection confidence.
        
        Args:
            rcs: Radar cross section
            distance: Range to target
            
        Returns:
            Confidence value 0-1
        """
        # Signal-to-noise ratio model
        snr = rcs / (distance / 1000.0)  # Simplified
        
        # Sigmoid confidence
        confidence = 1.0 / (1.0 + np.exp(-5 * (snr - 0.5)))
        
        return float(np.clip(confidence, 0, 1))


class InfraredSeeker:
    """
    Infrared heat-seeking sensor.
    """
    
    def __init__(
        self,
        range: float = 5000.0,
        lock_angle_deg: float = 10.0,
        noise_std: float = 0.05,
        ir_signature: float = 1.0
    ):
        """
        Initialize IR seeker.
        
        Args:
            range: Maximum detection range
            lock_angle_deg: Lock cone angle in degrees
            noise_std: Standard deviation for noise
            ir_signature: Base IR signature strength
        """
        self.range = range
        self.lock_angle = np.deg2rad(lock_angle_deg)
        self.noise_std = noise_std
        self.ir_signature = ir_signature
        
        logger.debug(f"IR Seeker: range={range}, lock_angle={lock_angle_deg}°")
    
    def get_signal(
        self,
        owner_entity,
        physics_world,
        all_entities: List
    ) -> Dict:
        """
        Get IR signal from entities.
        
        Args:
            owner_entity: Entity that owns this sensor
            physics_world: PhysicsWorld instance
            all_entities: List of all entities
            
        Returns:
            Dict with best lock information or None
        """
        if not owner_entity.is_alive:
            return {'locked': False, 'target': None}
        
        owner_pos = owner_entity.position
        
        # Get forward direction
        if hasattr(owner_entity, 'body_id') and owner_entity.body_id is not None:
            forward = physics_world.get_forward_vector(owner_entity.body_id)
        else:
            forward = np.array([1, 0, 0])
        
        best_lock = None
        best_confidence = 0.0
        
        for entity in all_entities:
            if entity.id == owner_entity.id:
                continue
            
            if not hasattr(entity, 'is_alive') or not entity.is_alive:
                continue
            
            # Direction to target
            to_target = entity.position - owner_pos
            distance = np.linalg.norm(to_target)
            
            if distance > self.range or distance < 1.0:
                continue
            
            target_dir = to_target / distance
            
            # Check if within lock cone
            angle = np.arccos(np.clip(np.dot(forward, target_dir), -1, 1))
            
            if angle > self.lock_angle / 2:
                continue
            
            # Compute IR signature
            signature = self._compute_ir_signature(entity, distance)
            
            # Add noise
            signature_noisy = signature + np.random.normal(0, self.noise_std)
            
            # Confidence based on signature strength
            confidence = self._compute_confidence(signature_noisy)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_lock = {
                    'id': entity.id,
                    'distance': distance,
                    'angle': angle,
                    'signal_strength': signature_noisy,
                    'confidence': confidence,
                    'position': entity.position.tolist()
                }
        
        # Threshold for lock
        if best_confidence > 0.3:
            return {'locked': True, 'target': best_lock}
        else:
            return {'locked': False, 'target': None}
    
    def _compute_ir_signature(self, entity, distance: float) -> float:
        """
        Compute IR signature (simplified model).
        
        Args:
            entity: Target entity
            distance: Range to target
            
        Returns:
            IR signature value
        """
        # Base signature based on entity type
        if hasattr(entity, '__class__'):
            if entity.__class__.__name__ == 'Attacker':
                # Higher if afterburner/thrust is active
                if hasattr(entity, 'current_thrust') and entity.current_thrust > 0:
                    base_sig = self.ir_signature * 2.0
                else:
                    base_sig = self.ir_signature
            elif entity.__class__.__name__ == 'Interceptor':
                base_sig = self.ir_signature * 3.0  # Hotter
            elif entity.__class__.__name__ == 'Defender':
                base_sig = self.ir_signature * 0.5
            else:
                base_sig = self.ir_signature
        else:
            base_sig = self.ir_signature
        
        # Distance falloff (inverse square)
        distance_falloff = (1000.0 / max(distance, 100)) ** 2
        
        return base_sig * distance_falloff
    
    def _compute_confidence(self, signal: float) -> float:
        """
        Compute lock confidence from signal strength.
        
        Args:
            signal: IR signal strength
            
        Returns:
            Confidence 0-1
        """
        confidence = 1.0 / (1.0 + np.exp(-10 * (signal - 0.3)))
        return float(np.clip(confidence, 0, 1))
