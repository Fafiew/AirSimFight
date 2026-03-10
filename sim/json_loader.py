import json
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except Exception:  # pragma: no cover
    Draft7Validator = None

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["world", "attackers", "defenders", "targets"],
}


def _manual_validate(data):
    if set(data.keys()) != {"world", "attackers", "defenders", "targets"}:
        raise ValueError("Top-level keys must be exactly world, attackers, defenders, targets")
    if set(data["world"].keys()) != {"size"}:
        raise ValueError("world must contain only size")
    for k in ["attackers", "defenders", "targets"]:
        for e in data[k]:
            if set(e.keys()) != {"id", "position"}:
                raise ValueError(f"{k} entities must contain only id and position")
            if not isinstance(e["position"], list) or len(e["position"]) != 3:
                raise ValueError("position must be length-3 list")


def load_positions(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    if Draft7Validator is None:
        _manual_validate(data)
    else:
        # strict schema for extra-key rejection
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["world", "attackers", "defenders", "targets"],
            "properties": {
                "world": {"type": "object", "additionalProperties": False, "required": ["size"], "properties": {"size": {"type": "number"}}},
                "attackers": {"type": "array", "items": {"$ref": "#/definitions/entity"}},
                "defenders": {"type": "array", "items": {"$ref": "#/definitions/entity"}},
                "targets": {"type": "array", "items": {"$ref": "#/definitions/entity"}},
            },
            "definitions": {
                "entity": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "position"],
                    "properties": {"id": {"type": "string"}, "position": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}},
                }
            },
        }
        errors = sorted(Draft7Validator(schema).iter_errors(data), key=lambda e: e.path)
        if errors:
            raise ValueError("Invalid visualization JSON: " + "; ".join(e.message for e in errors))
    return {k: [{"id": e["id"], "position": e["position"]} for e in data[k]] for k in ["attackers", "defenders", "targets"]}
