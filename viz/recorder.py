"""
Video recorder for AirSimFight.
Captures frames and encodes to MP4 using FFmpeg.
"""

import os
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class VideoRecorder:
    """
    Records simulation frames to MP4 video.
    """
    
    def __init__(
        self,
        output_path: str = "output.mp4",
        fps: int = 30,
        frames_dir: Optional[str] = None,
        cleanup_frames: bool = True
    ):
        """
        Initialize recorder.
        
        Args:
            output_path: Output MP4 file path
            fps: Frames per second
            frames_dir: Directory for temporary frame PNGs
            cleanup_frames: Whether to delete frames after encoding
        """
        self.output_path = output_path
        self.fps = fps
        self.cleanup_frames = cleanup_frames
        
        # Default frames directory
        if frames_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frames_dir = f"frames_{timestamp}"
        
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_count = 0
        self.is_recording = False
        
        # Check FFmpeg
        self.ffmpeg_available = self._check_ffmpeg()
        
        logger.info(f"VideoRecorder initialized: output={output_path}, fps={fps}")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not found. Video export will not be available.")
            logger.warning("Install with: sudo apt-get install ffmpeg (Ubuntu) or brew install ffmpeg (macOS)")
            return False
    
    def start_recording(self) -> None:
        """Start recording frames."""
        self.is_recording = True
        logger.info("Recording started")
    
    def stop_recording(self) -> None:
        """Stop recording frames."""
        self.is_recording = False
        logger.info("Recording stopped")
    
    def capture_frame(self, frame_data: bytes) -> None:
        """
        Capture a single frame.
        
        Args:
            frame_data: Raw frame bytes (PNG format expected)
        """
        if not self.is_recording:
            return
        
        frame_path = self.frames_dir / f"frame_{self.frame_count:06d}.png"
        
        with open(frame_path, 'wb') as f:
            f.write(frame_data)
        
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            logger.debug(f"Captured {self.frame_count} frames")
    
    def save_frame_numpy(self, image_array, channel: str = 'RGB') -> None:
        """
        Save frame from numpy array.
        
        Args:
            image_array: Numpy array (H, W, C)
            channel: Channel order ('RGB' or 'BGR')
        """
        if not self.is_recording:
            return
        
        try:
            from PIL import Image
            import numpy as np
            
            frame_path = self.frames_dir / f"frame_{self.frame_count:06d}.png"
            
            # Convert array to PIL Image
            if channel == 'BGR':
                image_array = image_array[..., ::-1]
            
            img = Image.fromarray(image_array)
            img.save(frame_path)
            
            self.frame_count += 1
            
        except ImportError:
            logger.warning("PIL not available for frame capture")
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
    
    def encode_video(self) -> bool:
        """
        Encode captured frames to MP4.
        
        Returns:
            True if successful
        """
        if not self.ffmpeg_available:
            logger.error("FFmpeg not available for video encoding")
            return False
        
        if self.frame_count == 0:
            logger.warning("No frames captured")
            return False
        
        logger.info(f"Encoding {self.frame_count} frames to {self.output_path}")
        
        # Build FFmpeg command
        input_pattern = str(self.frames_dir / "frame_%06d.png")
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-framerate', str(self.fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            self.output_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Video saved to {self.output_path}")
            
            # Cleanup frames if requested
            if self.cleanup_frames:
                shutil.rmtree(self.frames_dir)
                logger.debug("Cleaned up frame files")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg encoding failed: {e}")
            logger.error(f"FFmpeg output: {e.stderr}")
            return False
    
    def capture_and_encode(self, frame_data: bytes) -> bool:
        """
        Capture frame and immediately encode video.
        Convenience method for single-frame capture.
        
        Args:
            frame_data: Frame bytes
            
        Returns:
            True if successful
        """
        self.capture_frame(frame_data)
        return self.encode_video()
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.frames_dir.exists() and self.cleanup_frames:
            try:
                shutil.rmtree(self.frames_dir)
                logger.debug("Cleaned up frames directory")
            except Exception as e:
                logger.warning(f"Error cleaning up frames: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


def check_ffmpeg() -> bool:
    """
    Check if FFmpeg is available.
    
    Returns:
        True if FFmpeg is installed
    """
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_instructions() -> str:
    """Get FFmpeg installation instructions for current OS."""
    import platform
    
    system = platform.system().lower()
    
    if system == 'linux':
        return "sudo apt-get install ffmpeg"
    elif system == 'darwin':
        return "brew install ffmpeg"
    elif system == 'windows':
        return "Download from https://ffmpeg.org/download.html or use chocolatey: choco install ffmpeg"
    else:
        return "Download from https://ffmpeg.org/download.html"


# CLI for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if check_ffmpeg():
        print("FFmpeg is available")
    else:
        print("FFmpeg is NOT available")
        print(f"Install with: {install_instructions()}")
