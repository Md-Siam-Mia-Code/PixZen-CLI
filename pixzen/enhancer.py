# pixzen/enhancer.py
from __future__ import annotations
import cv2
import numpy as np
import torch
import onnxruntime as ort
import warnings
import subprocess
import json
import threading
import shutil
import platform
import imageio.v2 as imageio
import time
import sys
import itertools
from pathlib import Path
from tqdm import tqdm
from typing import Any, Tuple
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.utils.download_util import load_file_from_url

from .utils import HardwareManager, get_realesrgan_model_details, GFPGAN_MODELS

PACKAGE_DIR = Path(__file__).parent


class ProcessingIndicator:
    def __init__(self, message: str = "Processing..."):
        self.message = message
        self.spinner = itertools.cycle(["⠇", "⠏", "⠋", "⠙", "⠸", "⠴", "⠦", "⠇"])
        self.done = False
        self.thread = threading.Thread(target=self._spin)

    def _spin(self):
        while not self.done:
            sys.stdout.write(f"\r{next(self.spinner)} {self.message} ")
            sys.stdout.flush()
            time.sleep(0.1)

    def __enter__(self):
        self.done = False
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.done = True
        self.thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 5) + "\r")
        sys.stdout.flush()


def _drain_pipe(pipe, description: str) -> None:
    try:
        with pipe:
            for line in iter(pipe.readline, b""):
                pass
    except Exception as e:
        print(f"Error reading from {description} pipe: {e}")


class PixZenEnhancer:
    def __init__(self, args: Any):
        self.args = args
        self.hw = HardwareManager()
        self.model_details = get_realesrgan_model_details(args.realesrgan)
        if not self.model_details:
            raise ValueError(f"Model alias {args.realesrgan} is not supported.")
        self.netscale = self.model_details["scale"]
        self._upsampler: RealESRGANer | ONNXRealESRGANer | None = None
        self._face_enhancer: GFPGANer | None = None

    def prepare_models(self) -> None:

        # Check and prepare the main Real-ESRGAN model
        realesrgan_model_info = get_realesrgan_model_details(self.args.realesrgan)
        use_onnx = self.hw.upscaler_backend == "onnx"
        self._get_model_path(
            realesrgan_model_info["url"],
            realesrgan_model_info["filename"],
            onnx=use_onnx,
        )

        # Check and prepare the GFPGAN model if it's going to be used
        if self.args.gfpgan:
            gfpgan_url = GFPGAN_MODELS[self.args.gfpgan]
            self._get_model_path(gfpgan_url, "GFPGANv1.3")

    @property
    def upsampler(self) -> RealESRGANer | ONNXRealESRGANer:
        if self._upsampler is None:
            if self.hw.upscaler_backend == "pytorch":
                model = self.model_details["model"]()
                model_path = self._get_model_path(
                    self.model_details["url"], self.model_details["filename"]
                )
                self._upsampler = RealESRGANer(
                    scale=self.netscale,
                    model_path=str(model_path),
                    model=model,
                    tile=self.args.tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True,
                    device=torch.device(self.hw.upscaler_device_name),
                )
            else:
                onnx_path = self._get_model_path(
                    self.model_details["url"], self.model_details["filename"], onnx=True
                )
                self._upsampler = ONNXRealESRGANer(onnx_path, self.hw.onnx_providers)
        return self._upsampler

    @property
    def face_enhancer(self) -> GFPGANer | None:
        if not self.args.gfpgan:
            return None
        if self._face_enhancer is None:
            model_url = GFPGAN_MODELS[self.args.gfpgan]
            model_path = self._get_model_path(model_url, "GFPGANv1.3")
            self._face_enhancer = GFPGANer(
                model_path=str(model_path),
                upscale=self.args.outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.upsampler,
                device=self.hw.face_enhancer_device,
            )
        return self._face_enhancer

    def _get_model_path(self, url: str, filename_stem: str, onnx: bool = False) -> Path:
        model_dir = Path.home() / ".cache" / "pixzen" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        pth_path = model_dir / f"{filename_stem}.pth"
        onnx_path = model_dir / f"{filename_stem}.onnx"
        if not pth_path.exists():
            print(f"Downloading model: {pth_path.name}...")
            load_file_from_url(
                url=url,
                model_dir=str(model_dir),
                progress=True,
                file_name=str(pth_path.name),
            )
        if onnx:
            if not onnx_path.exists():
                self._export_to_onnx(pth_path, onnx_path)
            return onnx_path
        return pth_path

    def _export_to_onnx(self, pth_path: Path, onnx_path: Path) -> None:
        # Since this might be called during pre-flight, we need the model details here too.
        realesrgan_model_info = get_realesrgan_model_details(self.args.realesrgan)
        print(
            f"--- Performing one-time export of {pth_path.name} to ONNX format... ---"
        )
        model = realesrgan_model_info["model"]()
        loadnet = torch.load(str(pth_path), map_location="cpu")
        keyname = "params_ema" if "params_ema" in loadnet else "params"
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        dummy_input = torch.randn(1, 3, 64, 64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                opset_version=13,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size", 2: "height", 3: "width"},
                    "output": {0: "batch_size", 2: "height", 3: "width"},
                },
            )
        print(f"--- ONNX model saved to {onnx_path} ---")

    def enhance_image(self, img_path: Path) -> None:
        with ProcessingIndicator(f"Enhancing {img_path.name}"):
            try:
                try:
                    img_raw = imageio.imread(str(img_path))
                    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
                        img = cv2.cvtColor(img_raw, cv2.COLOR_RGBA2BGRA)
                    else:
                        img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
                except Exception:
                    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(
                        f"\r⚠️ Warning: Could not read image {img_path.name}, skipping."
                    )
                    return
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                if self.face_enhancer:
                    _, _, output = self.face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True
                    )
                else:
                    output, _ = self.upsampler.enhance(img, outscale=self.args.outscale)

                if self.args.ext == "auto":
                    output_ext = img_path.suffix.lower()
                    if output_ext not in [".jpg", ".jpeg", ".png"]:
                        output_ext = ".png"
                else:
                    output_ext = f".{self.args.ext}"

                save_path = (
                    self.args.output / f"{self.args.prefix}_{img_path.stem}{output_ext}"
                )
                cv2.imwrite(str(save_path), output)
            except Exception as error:
                print(f"\r❌ Error processing '{img_path.name}': {error}")
                return
        print(
            f"✅ Enhanced {img_path.name} -> {self.args.output.name}/{save_path.name}"
        )

    def enhance_video(self, video_path: Path) -> None:
        ffmpeg_bin, ffprobe_bin = self._get_ffmpeg_paths()
        if not ffmpeg_bin or not ffprobe_bin:
            return
        try:
            info = self._get_video_info(video_path, ffprobe_bin)
            w, h, fps, num_frames, has_audio = info.values()
        except Exception as e:
            print(f"FATAL: Failed to get video info. Error: {e}")
            return
        out_w, out_h = int(w * self.args.outscale), int(h * self.args.outscale)
        output_path = self.args.output / f"{self.args.prefix}_{video_path.stem}.mp4"
        reader_process = self._create_ffmpeg_reader(ffmpeg_bin, video_path)
        writer_process = self._create_ffmpeg_writer(
            ffmpeg_bin, output_path, out_w, out_h, fps, has_audio, video_path
        )
        threading.Thread(
            target=_drain_pipe, args=(reader_process.stderr, "FFMPEG_READER")
        ).start()
        threading.Thread(
            target=_drain_pipe, args=(writer_process.stderr, "FFMPEG_WRITER")
        ).start()
        bytes_per_frame = w * h * 3
        desc = f"Upscaling {video_path.name}"
        with tqdm(total=num_frames, unit="frame", desc=desc[:40], ncols=100) as pbar:
            while True:
                frame_bytes = reader_process.stdout.read(bytes_per_frame)
                if not frame_bytes:
                    break
                frame = np.frombuffer(frame_bytes, np.uint8).reshape([h, w, 3])
                if self.face_enhancer:
                    _, _, output_frame = self.face_enhancer.enhance(
                        frame,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                    )
                else:
                    output_frame, _ = self.upsampler.enhance(
                        frame, outscale=self.args.outscale
                    )
                try:
                    writer_process.stdin.write(output_frame.tobytes())
                except (IOError, BrokenPipeError):
                    print(f"Error writing to ffmpeg stdin.")
                    break
                pbar.update(1)
        writer_process.stdin.close()
        reader_process.stdout.close()
        writer_process.wait()
        reader_process.wait()
        print(f"✅ Enhanced video -> {output_path}")

    def _get_ffmpeg_paths(self) -> Tuple[str | None, str | None]:
        ffmpeg_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
        ffprobe_name = "ffprobe.exe" if platform.system() == "Windows" else "ffprobe"
        ffmpeg_path, ffprobe_path = shutil.which(ffmpeg_name), shutil.which(
            ffprobe_name
        )
        if not ffmpeg_path:
            bundled_ffmpeg = PACKAGE_DIR / "vendor" / "ffmpeg" / ffmpeg_name
            if bundled_ffmpeg.is_file():
                ffmpeg_path = str(bundled_ffmpeg)
        if not ffprobe_path:
            bundled_ffprobe = PACKAGE_DIR / "vendor" / "ffmpeg" / ffprobe_name
            if bundled_ffprobe.is_file():
                ffprobe_path = str(bundled_ffprobe)
        if not ffmpeg_path:
            print(f"FATAL: Could not find '{ffmpeg_name}'.")
        if not ffprobe_path:
            print(f"FATAL: Could not find '{ffprobe_name}'.")
        return ffmpeg_path, ffprobe_path

    def _get_video_info(self, video_path: Path, ffprobe_path: str) -> Dict[str, Any]:
        command = [
            ffprobe_path,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        video_stream = next(
            s for s in probe_data["streams"] if s["codec_type"] == "video"
        )
        fps_str, num_frames = video_stream.get("avg_frame_rate", "0/0"), int(
            video_stream.get("nb_frames", 0)
        )
        fps = eval(fps_str) if "/" in fps_str else float(fps_str)
        if num_frames == 0:
            num_frames = int(float(probe_data["format"].get("duration", 0)) * fps)
        return {
            "width": int(video_stream["width"]),
            "height": int(video_stream["height"]),
            "fps": fps,
            "num_frames": num_frames,
            "has_audio": any(s["codec_type"] == "audio" for s in probe_data["streams"]),
        }

    def _create_ffmpeg_reader(
        self, ffmpeg_bin: str, video_path: Path
    ) -> subprocess.Popen:
        command = [
            ffmpeg_bin,
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-loglevel",
            "warning",
            "pipe:1",
        ]
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _create_ffmpeg_writer(
        self,
        ffmpeg_bin: str,
        output_path: Path,
        w: int,
        h: int,
        fps: float,
        has_audio: bool,
        audio_source: Path,
    ) -> subprocess.Popen:
        command = [
            ffmpeg_bin,
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
        ]
        if has_audio:
            command.extend(
                [
                    "-i",
                    str(audio_source),
                    "-c:a",
                    "copy",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                ]
            )
        command.extend(
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-loglevel",
                "warning",
                "-y",
                str(output_path),
            ]
        )
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


class ONNXRealESRGANer:
    def __init__(self, onnx_path: Path, providers: list[str]):
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.scale = 4

    def enhance(self, img: np.ndarray, outscale: float) -> Tuple[np.ndarray, None]:
        img = img.astype(np.float32) / 255.0
        has_alpha = img.shape[2] == 4
        if has_alpha:
            alpha = img[:, :, 3]
            img = img[:, :, :3]
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        input_tensor = np.expand_dims(img, axis=0)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: input_tensor})[0]
        result = np.squeeze(result, axis=0).clip(0, 1)
        output_img = np.transpose(result, (1, 2, 0))[:, :, [2, 1, 0]] * 255.0
        if has_alpha:
            h, w, _ = output_img.shape
            alpha_resized = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            output_img = cv2.merge(
                (output_img.astype(np.uint8), (alpha_resized * 255).astype(np.uint8))
            )
        return output_img.astype(np.uint8), None
