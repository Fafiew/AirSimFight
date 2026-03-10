#!/usr/bin/env bash
python -m rl.trainer_pretrain --config config/default.yaml --device cpu --timesteps 1000 --output models/attacker_pretrained --safety-check --force
