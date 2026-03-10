import shutil
import subprocess
from pathlib import Path


def export_mp4(output: str, fps: int = 30, cleanup: bool = True):
    frames = Path("frames")
    frames.mkdir(exist_ok=True)
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", "frames/frame_%06d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", output]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print("FFmpeg not found; cannot export MP4.")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e.stderr}")
    if cleanup and frames.exists():
        shutil.rmtree(frames)
