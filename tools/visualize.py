#!/usr/bin/env python3
"""
Visualization tool for AirSimFight.
Displays scenarios using Panda3D or prints dependency info.
"""

import argparse
import logging
import sys
from pathlib import Path
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.json_loader import load_positions, load_scenario
from sim.environment import AirCombatEnv
from sim.utils import load_config, setup_logging
from viz.renderer import create_renderer, check_panda3d
from viz.recorder import VideoRecorder

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Visualize AirSimFight scenarios")
    parser.add_argument('--scenario', type=str, required=True,
                       help='Path to scenario JSON file')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Config file path')
    parser.add_argument('--export', type=str, default=None,
                       help='Export to MP4 file')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for export')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no visualization)')
    parser.add_argument('--no-sensors', action='store_true',
                       help='Hide sensor cones')
    parser.add_argument('--no-hud', action='store_true',
                       help='Hide HUD')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check Panda3D
    if not args.headless:
        if not check_panda3d():
            print("Panda3D not found. Please install Panda3D or run in headless mode.")
            print("Install with: pip install panda3d")
            sys.exit(1)
    
    # Load scenario
    logger.info(f"Loading scenario from {args.scenario}")
    try:
        scenario_data = load_positions(args.scenario)
    except Exception as e:
        logger.error(f"Error loading scenario: {e}")
        sys.exit(1)
    
    # Load config
    config = load_config(args.config)
    
    # Setup environment for visualization
    env = AirCombatEnv(
        config=config,
        device='cpu',
        headless=args.headless,
        pretrain_mode=False
    )
    
    # Load entities from scenario
    from sim.entities import Attacker, Defender, Target
    
    # Create entities from scenario
    env.attackers.clear()
    env.defenders.clear()
    env.targets.clear()
    env.entities.clear()
    
    for a in scenario_data['attackers']:
        attacker = Attacker(a['id'], a['position'], config)
        env.attackers[attacker.id] = attacker
        env.entities[attacker.id] = attacker
    
    for d in scenario_data['defenders']:
        defender = Defender(d['id'], d['position'], config)
        env.defenders[defender.id] = defender
        env.defenders[defender.id] = defender
        env.entities[defender.id] = defender
    
    for t in scenario_data['targets']:
        target = Target(t['id'], t['position'], value=100.0)
        env.targets[target.id] = target
        env.entities[target.id] = target
    
    logger.info(f"Loaded: {len(env.attackers)} attackers, {len(env.defenders)} defenders, {len(env.targets)} targets")
    
    if args.headless:
        # Headless mode - just print info
        print("\n" + "="*50)
        print("SCENARIO INFO (headless mode)")
        print("="*50)
        print(f"Attackers: {len(env.attackers)}")
        for a in env.attackers.values():
            print(f"  {a.id}: position={a.position}")
        print(f"\nDefenders: {len(env.defenders)}")
        for d in env.defenders.values():
            print(f"  {d.id}: position={d.position}")
        print(f"\nTargets: {len(env.targets)}")
        for t in env.targets.values():
            print(f"  {t.id}: position={t.position}")
        print("="*50)
        
        if args.export:
            print("\nExport not supported in headless mode")
        
        env.close()
        return
    
    # Create renderer
    renderer = create_renderer(
        env,
        headless=False,
        show_sensor_cones=not args.no_sensors,
        show_hud=not args.no_hud
    )
    
    if renderer is None:
        print("Failed to create renderer")
        env.close()
        sys.exit(1)
    
    # Setup recorder if exporting
    recorder = None
    if args.export:
        recorder = VideoRecorder(
            output_path=args.export,
            fps=args.fps
        )
        recorder.start_recording()
    
    # Run visualization
    logger.info("Starting visualization...")
    logger.info("Controls: p=pause, r=reset, v=toggle sensors, e=export, arrows=camera")
    
    # Main loop would go here
    # For now, just note that it would run
    
    env.close()
    logger.info("Visualization complete")


if __name__ == "__main__":
    main()
