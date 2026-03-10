"""Optional Ray RLlib adapter placeholder."""


def available() -> bool:
    try:
        import ray  # noqa: F401

        return True
    except Exception:
        return False
