import json
from pathlib import Path

import pytest

from sim.json_loader import load_positions


def test_load_positions_ok():
    data = load_positions("scenarios/example_visualization.json")
    assert isinstance(data["attackers"], list)
    assert data["attackers"][0]["id"] == "a1"


def test_reject_extra_entity_keys(tmp_path: Path):
    bad = {
        "world": {"size": 100},
        "attackers": [{"id": "a", "position": [0, 0, 0], "speed": 1}],
        "defenders": [],
        "targets": [],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_positions(str(p))
