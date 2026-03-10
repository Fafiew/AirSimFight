# AirSimFight Safety & Ethics Guidelines

## ⚠️ CRITICAL WARNING

**AirSimFight is strictly a simulation and research tool.** It is designed for academic research, reinforcement learning experimentation, and air combat simulation studies only.

## Prohibited Uses

The following uses of this software are **strictly prohibited**:

1. **Real-world weapons development** - Do NOT use this code to develop, maintain, or control real-world weapons, munitions, or operational interception systems.

2. **Hardware transfer** - Do NOT transfer trained models, parameters, or sensor configurations to real military hardware.

3. **Operational use** - Do NOT use this simulation for real-world military operations, tactics, or strategy development.

4. **Autonomous lethal systems** - Do NOT use this software to develop autonomous weapons systems that could target and kill humans.

## Best Practices

1. **Isolated computing** - Run the simulation on isolated/offline compute systems not connected to operational networks.

2. **Research only** - Use this tool only for academic research, AI/RL experimentation, and educational purposes.

3. **No export controls circumvention** - Do not use this software to circumvent any export controls or sanctions.

4. **Model transparency** - Document any trained models with clear statements that they are for simulation/research only.

## Safety Features

The code includes safety guardrails:

- `--safety-check` flag requires users to acknowledge safety guidelines before training
- Training operates in simulation only with no hardware interface
- Models are saved in standard formats that cannot be directly deployed to weapons systems

## Ethical AI Research

When publishing research using this simulation:

1. Include clear statements that the work is simulation-based research
2. Do not present simulation results as applicable to real-world combat
3. Emphasize the research nature of the work in all publications
4. Consider the broader implications of AI in combat simulation

## Liability

The authors and contributors of AirSimFight provide this software "as is" without warranty. Users assume all responsibility for compliance with applicable laws and regulations.

---

**By using this software, you agree to these terms and acknowledge this is a research tool only.**
