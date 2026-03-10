#!/bin/bash
# Example: Run pretraining

python -m rl.trainer_pretrain \
    --config config/default.yaml \
    --device cpu \
    --timesteps 1000000 \
    --output models/attacker_pretrained \
    --safety-check \
    --force
