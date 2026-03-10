#!/bin/bash
# Example: Run dual training

python -m rl.trainer_dual \
    --config config/default.yaml \
    --device cuda \
    --timesteps 2000000 \
    --attacker-model models/attacker_pretrained.zip \
    --fine-tune-attacker
