import argparse

from sim.json_loader import load_positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--export", default=None)
    args = ap.parse_args()

    try:
        from panda3d.core import NodePath  # noqa: F401
    except Exception:
        print("Panda3D not found. Please install Panda3D or run in headless mode.")
        return

    data = load_positions(args.scenario)
    print(f"Loaded scenario with {len(data['attackers'])} attackers, {len(data['defenders'])} defenders")
    if args.export:
        from viz.recorder import export_mp4

        export_mp4(args.export)


if __name__ == "__main__":
    main()
