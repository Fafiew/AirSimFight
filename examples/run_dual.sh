#!/usr/bin/env bash
python -m rl.trainer_dual --config config/default.yaml --device cpu --timesteps 1000 --attacker-model models/attacker_pretrained.zip --fine_tune_attacker
