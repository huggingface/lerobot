"""
GPU-accelerated video encoding utilities for LeRobot.

This module provides GPU-accelerated video encoding capabilities using:
- NVIDIA NVENC (H.264, HEVC, AV1)
- Intel Quick Sync (H.264, HEVC)
- AMD VCE (H.264, HEVC)
- Software fallback (libx264, libx265, libsvtav1)
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUEncoderConfig:
    """Configuration for GPU-accelerated video encoding."""
    
    def __init__(
        self,
        encoder_type: str = "auto",  # "nvenc", "qsv", "vce", "software", "auto"
        codec: str = "h264",  # "h264", "hevc", "av1"
        preset: str = "fast",  # "fast", "medium", "slow", "hq"
        quality: int = 23,  # Lower = better quality (for NVENC: 0-51, for x264: 0-51)
        bitrate: Optional[str] = None,  # e.g., "5M", "10M"
        gpu_id: int = 0,  # GPU device ID
        enable_logging: bool = True
    ):
        self.encoder_type = encoder_type
        self.codec = codec
        self.preset = preset
        self.quality = quality
        self.bitrate = bitrate
        self.gpu_id = gpu_id
        self.enable_logging = enable_logging
    
    def __str__(self) -> str:
        return f"GPUEncoderConfig(encoder={self.encoder_type}, codec={self.codec}, preset={self.preset}, quality={self.quality})"


class GPUVideoEncoder:
    """GPU-accelerated video encoder using FFmpeg."""
    
    def __init__(self, config: GPUEncoderConfig):
        self.config = config
        self._detected_encoders = self._detect_available_encoders()
        
        if self.config.enable_logging:
            logger.info(f"GPUVideoEncoder initialized with config: {config}")
            logger.info(f"Available encoders: {list(self._detected_encoders.keys())}")
    
    def _detect_available_encoders(self) -> Dict[str, List[str]]:
        """Detect available hardware and software encoders."""
        encoders = {}
        
        try:
            # Get FFmpeg encoder list
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                raise FileNotFoundError("ffmpeg not found in PATH")
                
            result = subprocess.run(
                [ffmpeg_path, "-encoders"],
                capture_output=True,
                text=True,
                check=True
            )
            
            output = result.stdout
            
            # NVIDIA NVENC encoders
            if "h264_nvenc" in output:
                encoders["nvenc"] = ["h264"]
                if "hevc_nvenc" in output:
                    encoders["nvenc"].append("hevc")
                if "av1_nvenc" in output:
                    encoders["nvenc"].append("av1")
            
            # Intel Quick Sync encoders
            if "h264_qsv" in output:
                encoders["qsv"] = ["h264"]
                if "hevc_qsv" in output:
                    encoders["qsv"].append("hevc")
            
            # AMD VCE encoders
            if "h264_amf" in output:
                encoders["vce"] = ["h264"]
                if "hevc_amf" in output:
                    encoders["vce"].append("hevc")
            
            # Software encoders
            encoders["software"] = []
            if "libx264" in output:
                encoders["software"].append("h264")
            if "libx265" in output:
                encoders["software"].append("hevc")
            if "libsvtav1" in output:
                encoders["software"].append("av1")
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to detect encoders: {e}")
            encoders = {"software": ["h264"]}  # Fallback
        
        return encoders
    
    def _select_encoder(self) -> Tuple[str, str]:
        """Select the best available encoder for the requested configuration."""
        if self.config.encoder_type == "auto":
            # Try hardware encoders first, then software
            for encoder_type in ["nvenc", "qsv", "vce", "software"]:
                if encoder_type in self._detected_encoders:
                    if self.config.codec in self._detected_encoders[encoder_type]:
                        return encoder_type, self.config.codec
            
            # Fallback to H.264 if requested codec not available
            for encoder_type in ["nvenc", "qsv", "vce", "software"]:
                if encoder_type in self._detected_encoders:
                    if "h264" in self._detected_encoders[encoder_type]:
                        logger.warning(f"Requested codec {self.config.codec} not available, falling back to H.264")
                        return encoder_type, "h264"
        
        elif self.config.encoder_type in self._detected_encoders:
            if self.config.codec in self._detected_encoders[self.config.encoder_type]:
                return self.config.encoder_type, self.config.codec
            elif "h264" in self._detected_encoders[self.config.encoder_type]:
                logger.warning(f"Requested codec {self.config.codec} not available for {self.config.encoder_type}, falling back to H.264")
                return self.config.encoder_type, "h264"
        
        # Final fallback to software H.264
        logger.warning(f"No suitable encoder found, falling back to software H.264")
        return "software", "h264"
    
    def _build_ffmpeg_command(
        self,
        input_dir: Path,
        output_path: Path,
        fps: int,
        encoder_type: str,
        codec: str
    ) -> List[str]:
        """Build FFmpeg command for GPU-accelerated encoding."""
        
        # Get FFmpeg path
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise FileNotFoundError("ffmpeg not found in PATH")
        
        # Base command
        cmd = [ffmpeg_path, "-y"]  # -y to overwrite output files
        
        # Input settings
        cmd.extend([
            "-framerate", str(fps),
            "-i", str(input_dir / "frame_%06d.png")
        ])
        
        # Video codec settings
        if encoder_type == "nvenc":
            cmd.extend([
                "-c:v", f"{codec}_nvenc",
                "-preset", self.config.preset,
                "-rc", "vbr",  # Variable bitrate
                "-cq", str(self.config.quality),
                "-b:v", self.config.bitrate or "5M",
                "-maxrate", "10M",
                "-bufsize", "10M",
                "-gpu", str(self.config.gpu_id)
            ])
        
        elif encoder_type == "qsv":
            cmd.extend([
                "-c:v", f"{codec}_qsv",
                "-preset", self.config.preset,
                "-global_quality", str(self.config.quality),
                "-b:v", self.config.bitrate or "5M"
            ])
        
        elif encoder_type == "vce":
            cmd.extend([
                "-c:v", f"{codec}_amf",
                "-quality", self.config.preset,
                "-rc", "vbr_peak",
                "-qp_i", str(self.config.quality),
                "-qp_p", str(self.config.quality),
                "-b:v", self.config.bitrate or "5M"
            ])
        
        else:  # software
            if codec == "h264":
                cmd.extend([
                    "-c:v", "libx264",
                    "-preset", self.config.preset,
                    "-crf", str(self.config.quality),
                    "-b:v", self.config.bitrate or "5M"
                ])
            elif codec == "hevc":
                cmd.extend([
                    "-c:v", "libx265",
                    "-preset", self.config.preset,
                    "-crf", str(self.config.quality),
                    "-b:v", self.config.bitrate or "5M"
                ])
            elif codec == "av1":
                cmd.extend([
                    "-c:v", "libsvtav1",
                    "-preset", str(self._map_preset_to_svt(self.config.preset)),
                    "-crf", str(self.config.quality),
                    "-b:v", self.config.bitrate or "5M"
                ])
        
        # Output settings
        cmd.extend([
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            str(output_path)
        ])
        
        return cmd
    
    def _map_preset_to_svt(self, preset: str) -> int:
        """Map preset names to SVT-AV1 preset numbers."""
        preset_map = {
            "fast": 8,
            "medium": 6,
            "slow": 4,
            "hq": 2
        }
        return preset_map.get(preset, 6)
    
    def encode_video(
        self,
        input_dir: Path,
        output_path: Path,
        fps: int,
        timeout: int = 300
    ) -> bool:
        """
        Encode a video using GPU acceleration.
        
        Args:
            input_dir: Directory containing frame images
            output_path: Output video file path
            fps: Frames per second
            
        Returns:
            True if encoding succeeded, False otherwise
        """
        
        # Ensure input directory exists
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return False
        
        # Check for frame files
        frame_files = list(input_dir.glob("frame_*.png"))
        if not frame_files:
            logger.error(f"No frame files found in {input_dir}")
            return False
        
        # Select encoder
        encoder_type, codec = self._select_encoder()
        
        if self.config.enable_logging:
            logger.info(f"Encoding video with {encoder_type} {codec} encoder")
            logger.info(f"Input: {input_dir} ({len(frame_files)} frames)")
            logger.info(f"Output: {output_path}")
        
        # Build FFmpeg command
        cmd = self._build_ffmpeg_command(input_dir, output_path, fps, encoder_type, codec)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run FFmpeg
            if self.config.enable_logging:
                logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout
            )
            
            if self.config.enable_logging:
                logger.info(f"Video encoding completed successfully: {output_path}")
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg encoding timed out after {timeout} seconds")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg encoding failed: {e}")
            if e.stderr:
                logger.error(f"FFmpeg stderr: {e.stderr}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error during encoding: {e}")
            return False
    
    def get_encoder_info(self) -> Dict[str, any]:
        """Get information about available encoders and current configuration."""
        encoder_type, codec = self._select_encoder()
        
        return {
            "available_encoders": self._detected_encoders,
            "selected_encoder": encoder_type,
            "selected_codec": codec,
            "config": {
                "encoder_type": self.config.encoder_type,
                "codec": self.config.codec,
                "preset": self.config.preset,
                "quality": self.config.quality,
                "gpu_id": self.config.gpu_id
            }
        }


def create_gpu_encoder_config(
    encoder_type: str = "auto",
    codec: str = "h264",
    preset: str = "fast",
    quality: int = 23,
    bitrate: Optional[str] = None,
    gpu_id: int = 0,
    enable_logging: bool = True
) -> GPUEncoderConfig:
    """Create a GPU encoder configuration."""
    return GPUEncoderConfig(
        encoder_type=encoder_type,
        codec=codec,
        preset=preset,
        quality=quality,
        bitrate=bitrate,
        gpu_id=gpu_id,
        enable_logging=enable_logging
    )


def test_gpu_encoding():
    """Test GPU encoding capabilities."""
    print("=" * 80)
    print("GPU ENCODING CAPABILITIES TEST")
    print("=" * 80)
    
    # Test different configurations
    configs = [
        create_gpu_encoder_config(encoder_type="auto", codec="h264", preset="fast"),
        create_gpu_encoder_config(encoder_type="nvenc", codec="h264", preset="fast"),
        create_gpu_encoder_config(encoder_type="nvenc", codec="hevc", preset="fast"),
        create_gpu_encoder_config(encoder_type="software", codec="h264", preset="fast"),
    ]
    
    for config in configs:
        print(f"\nTesting config: {config}")
        encoder = GPUVideoEncoder(config)
        info = encoder.get_encoder_info()
        
        print(f"  Available encoders: {info['available_encoders']}")
        print(f"  Selected encoder: {info['selected_encoder']}")
        print(f"  Selected codec: {info['selected_codec']}")
    
    print("\n" + "=" * 80)
    print("GPU ENCODING TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_gpu_encoding() 