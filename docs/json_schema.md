# AirSimFight JSON Schema

## Overview

Visualization JSON files in AirSimFight contain **only** position data for entities. They are used purely for visualization purposes and are **not** read by the training system.

## Schema Definition

### Root Object

```json
{
  "world": {...},
  "attackers": [...],
  "defenders": [...],
  "targets": [...]
}
```

### World Object

| Field | Type | Description |
|-------|------|-------------|
| `size` | number | World size in meters |

### Entity Arrays

Each entity array contains objects with only two fields:

#### Attackers

```json
{
  "attackers": [
    {"id": "string", "position": [x, y, z]}
  ]
}
```

- `id`: Unique identifier (string)
- `position`: 3D coordinates [x, y, z] in meters

#### Defenders

```json
{
  "defenders": [
    {"id": "string", "position": [x, y, z]}
  ]
}
```

- `id`: Unique identifier (string)
- `position`: 3D coordinates [x, y, z] in meters

#### Targets

```json
{
  "targets": [
    {"id": "string", "position": [x, y, z]}
  ]
}
```

- `id`: Unique identifier (string)
- `position`: 3D coordinates [x, y, z] in meters

## Validation Rules

1. **No extra fields**: Entity objects must contain ONLY `id` and `position`
2. **Array types**: Must be arrays of objects
3. **Position format**: Must be array of 3 numbers [x, y, z]
4. **Required keys**: `world`, `attackers`, `defenders`, `targets` must all be present

## Example Valid JSON

```json
{
  "world": {"size": 20000},
  "attackers": [
    {"id": "a1", "position": [5000, 0, 2000]},
    {"id": "a2", "position": [5200, 100, 2100]}
  ],
  "defenders": [
    {"id": "d1", "position": [100, 0, 0]}
  ],
  "targets": [
    {"id": "t1", "position": [0, 0, 0]}
  ]
}
```

## Example Invalid JSON

The following JSON is **invalid** because it contains extra fields:

```json
{
  "world": {"size": 20000},
  "attackers": [
    {
      "id": "a1",
      "position": [5000, 0, 2000],
      "velocity": [100, 0, 0],
      "health": 100
    }
  ]
}
```

This would be rejected by the loader because `velocity` and `health` are not allowed in visualization JSONs.

## Usage in Code

```python
from sim.json_loader import load_positions

# Load visualization data
data = load_positions("scenarios/example_visualization.json")

# Returns dict with:
# {
#   "attackers": [{"id": "a1", "position": [x, y, z]}, ...],
#   "defenders": [...],
#   "targets": [...]
# }
```

## Purpose

This schema ensures:
1. **Visualization-only**: JSON files cannot influence training dynamics
2. **Separation of concerns**: Training uses procedural generation, visualization uses pre-defined scenarios
3. **Security**: Prevents loading malicious entity configurations
