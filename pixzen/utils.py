# pixzen/utils.py
from __future__ import annotations
import torch
import onnxruntime as ort
import platform
import psutil
import subprocess
import shutil
from cpuinfo import get_cpu_info
from typing import Dict, Any, Literal
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Type definitions
Backend = Literal["pytorch", "onnx"]
Device = Literal["cuda", "dml", "coreml", "cpu"]

# --- NEW MODEL DEFINITIONS ---
REALESRGAN_MODELS: Dict[str, Dict[str, Any]] = {
    "RealESRGAN_x2plus": {
        "alias": "v1",
        "filename": "PixZen-SR-v1",
        "scale": 2,
        "model": lambda: RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        ),
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
    "RealESRGAN_x4plus": {
        "alias": "v2",
        "filename": "PixZen-SR-v2",
        "scale": 4,
        "model": lambda: RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        ),
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "alias": "Anime",
        "filename": "PixZen-v2-Anime",
        "scale": 4,
        "model": lambda: RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        ),
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    },
    "realesr-animevideov3": {
        "alias": "AnimeVideo",
        "filename": "PixZen-AnimeVideo",
        "scale": 4,
        "model": lambda: SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        ),
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    },
}

GFPGAN_MODELS: Dict[str, str] = {
    "v1": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    "v2": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
}

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".flv", ".avi", ".webm")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def get_realesrgan_model_details(alias: str) -> Dict[str, Any] | None:
    
    for model_info in REALESRGAN_MODELS.values():
        if model_info["alias"] == alias:
            return model_info
    return None


class HardwareManager:
    def __init__(self, gpu_id: int | None = None):
        self.upscaler_backend: Backend = "onnx"
        self.upscaler_device_name: Device | str = "cpu"
        self.face_enhancer_device: torch.device = torch.device("cpu")
        self.os_name: str = ""
        self.cpu_name: str = ""
        self.gpu_name: str = "N/A"
        self.total_ram: float = 0.0
        self._get_system_specs()
        self._detect_processing_devices(gpu_id)
        self._print_status_panel()

    def _run_shell_command(self, command: list[str]) -> str:
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    def _get_windows_specs(self) -> None:
        self.os_name = self._run_shell_command(
            [
                "powershell",
                "-Command",
                "(Get-CimInstance -ClassName Win32_OperatingSystem).Caption",
            ]
        )
        self.gpu_name = self._run_shell_command(
            [
                "powershell",
                "-Command",
                "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Name",
            ]
        )
        if not self.os_name:
            self.os_name = f"{platform.system()} {platform.release()}"

    def _get_linux_gpu_name(self) -> str:
        if shutil.which("lspci"):
            cmd_output = self._run_shell_command(
                ["sh", "-c", "lspci | grep -i 'VGA\\|3D'"]
            )
            if cmd_output:
                return cmd_output.split(":")[-1].strip()
        return "N/A"

    def _get_macos_gpu_name(self) -> str:
        cmd_output = self._run_shell_command(
            ["sh", "-c", "system_profiler SPDisplaysDataType | grep 'Chipset Model'"]
        )
        if cmd_output:
            return cmd_output.split(":")[-1].strip()
        return "N/A"

    def _get_system_specs(self) -> None:
        self.cpu_name = get_cpu_info().get("brand_raw", "Unknown CPU")
        self.total_ram = psutil.virtual_memory().total / (1024**3)
        if platform.system() == "Windows":
            self._get_windows_specs()
        elif platform.system() == "Linux":
            self.os_name = f"{platform.system()} {platform.release()}"
            self.gpu_name = self._get_linux_gpu_name()
        elif platform.system() == "Darwin":
            self.os_name = f"{platform.system()} {platform.release()}"
            self.gpu_name = self._get_macos_gpu_name()
        else:
            self.os_name = f"{platform.system()} {platform.release()}"
            self.gpu_name = "N/A"

    def _detect_processing_devices(self, gpu_id: int | None) -> None:
        if gpu_id is not None:
            if torch.cuda.is_available():
                self.upscaler_backend, self.upscaler_device_name = (
                    "pytorch",
                    f"cuda:{gpu_id}",
                )
                self.face_enhancer_device = torch.device(f"cuda:{gpu_id}")
                self.gpu_name = torch.cuda.get_device_name(gpu_id)
                return
            else:
                print(
                    f"⚠️ Warning: GPU ID {gpu_id} requested, but no CUDA-enabled GPU found. Auto-detecting..."
                )
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.upscaler_backend, self.upscaler_device_name = "pytorch", "cuda"
            self.face_enhancer_device = torch.device("cuda")
            self.gpu_name = torch.cuda.get_device_name(0)
            return
        available_providers = ort.get_available_providers()
        if "DmlExecutionProvider" in available_providers:
            self.upscaler_backend, self.upscaler_device_name = "onnx", "dml"
            self.face_enhancer_device = torch.device("cpu")
            return
        if "CoreMLExecutionProvider" in available_providers:
            self.upscaler_backend, self.upscaler_device_name = "onnx", "coreml"
            self.face_enhancer_device = torch.device("cpu")
            if not self.gpu_name or self.gpu_name == "N/A":
                self.gpu_name = "Apple Silicon"
            return
        self.upscaler_backend, self.upscaler_device_name = "onnx", "cpu"
        self.face_enhancer_device = torch.device("cpu")

    def _print_status_panel(self) -> None:
        print("┌" + "─" * 15 + " SYSTEM " + "─" * 25 + "┐")
        print(f"│ OS:            {self.os_name[:32]:<32}│")
        print(f"│ CPU:           {self.cpu_name[:32]:<32}│")
        print(f"│ GPU:           {self.gpu_name[:32]:<32}│")
        print(f"│ RAM:           {f'{self.total_ram:.1f} GB':<32}│")
        print("├" + "─" * 14 + " SETTINGS " + "─" * 24 + "┤")
        print(
            f"│ Upscaler:      {self.upscaler_backend.upper()} on {self.upscaler_device_name.upper():<23} │"
        )
        print(
            f"│ Face Enhancer: PyTorch on {str(self.face_enhancer_device).upper():<21}│"
        )
        print("└" + "─" * 48 + "┘")

    @property
    def onnx_providers(self) -> list[str]:
        if self.upscaler_device_name == "dml":
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if self.upscaler_device_name == "coreml":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        if self.upscaler_device_name.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
