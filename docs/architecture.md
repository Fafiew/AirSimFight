# Architecture
Two-phase training: attacker pretraining then dual-agent training with procedural spawning (1-10 attackers, 1-10 defenders).

Attacker reward defaults:
- +target_reward on target reach
- -destroyed_penalty when destroyed
- -fuel_cost_coeff * fuel_used
- small distance-to-target shaping
