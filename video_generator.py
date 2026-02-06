"""
TikTok-Style Video Generator — Generate 10-second videos from text prompts.

Two-stage pipeline powered by Wan 2.2:
  Stage 1: Text → Image  (SDXL generates a high-quality starting frame)
  Stage 2: Image → Video  (Wan 2.2 TI2V-5B animates the frame into a 10s video)

Alternatively supports direct Text → Video with Wan 2.2 TI2V-5B (--mode t2v).

Model stack:
  - Image Gen : stabilityai/stable-diffusion-xl-base-1.0 (~6.5 GB in fp16)
  - Video Gen : Wan-AI/Wan2.2-TI2V-5B-Diffusers  (~10 GB in bf16)

Requirements:
  pip install git+https://github.com/huggingface/diffusers
  pip install transformers accelerate sentencepiece ftfy protobuf

Usage:
  python video_generator.py "A cat skateboarding through a neon city"
  python video_generator.py --example 2 --preset tiktok_10s
  python video_generator.py --mode t2v --example 3
  python video_generator.py --list-presets
  python video_generator.py --list-examples
"""

import argparse
import gc
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from PIL import Image

from gpu_monitor import GPUPowerMonitor


# ---------------------------------------------------------------------------
# Video presets for Wan 2.2
# Resolutions: 480P (480×832) or 720P (704×1280 for TI2V-5B)
# Frame count formula: num_frames = 4 * k + 1 (required by Wan VAE)
# ---------------------------------------------------------------------------
PRESETS = {
    "tiktok_short": {
        "width": 480,
        "height": 832,
        "num_frames": 81,        # ~5 s @ 16 fps
        "fps": 16,
        "description": "Short TikTok clip (portrait 480P, ~5 s)",
    },
    "tiktok_10s": {
        "width": 480,
        "height": 832,
        "num_frames": 161,       # ~10 s @ 16 fps
        "fps": 16,
        "description": "Standard TikTok clip (portrait 480P, ~10 s)",
    },
    "landscape": {
        "width": 832,
        "height": 480,
        "num_frames": 81,        # ~5 s @ 16 fps
        "fps": 16,
        "description": "Landscape clip (480P, ~5 s)",
    },
    "landscape_10s": {
        "width": 832,
        "height": 480,
        "num_frames": 161,       # ~10 s @ 16 fps
        "fps": 16,
        "description": "Landscape clip (480P, ~10 s)",
    },
    "hd_tiktok": {
        "width": 704,
        "height": 1280,
        "num_frames": 81,        # ~5 s @ 16 fps
        "fps": 16,
        "description": "HD TikTok clip (portrait 720P, ~5 s)",
    },
    "hd_tiktok_10s": {
        "width": 704,
        "height": 1280,
        "num_frames": 161,       # ~10 s @ 16 fps
        "fps": 16,
        "description": "HD TikTok clip (portrait 720P, ~10 s)",
    },
}

# ---------------------------------------------------------------------------
# Example prompts — TikTok-style content ideas
# ---------------------------------------------------------------------------
EXAMPLE_PROMPTS = [
    # 0 – Dance / neon vibe
    (
        "A person dancing energetically in a neon-lit room with colorful purple "
        "and blue lights reflecting off the walls, dynamic camera movement follows "
        "the dancer, vibrant electric atmosphere, cinematic lighting, photorealistic"
    ),
    # 1 – Cute animal
    (
        "A golden retriever puppy wearing tiny sunglasses sitting on a surfboard "
        "riding a small wave at the beach, close-up shot, the puppy looks excited, "
        "bright sunny day, turquoise ocean water splashing gently, sharp focus"
    ),
    # 2 – Satisfying / ASMR
    (
        "Satisfying top-down footage of colorful acrylic paint being poured and "
        "swirled in slow motion on a white canvas, vibrant pink teal and gold "
        "colors merge into mesmerizing patterns, soft studio lighting, macro"
    ),
    # 3 – Nature timelapse
    (
        "A breathtaking timelapse of a sunset over the ocean, golden and orange "
        "clouds move across the sky while waves crash rhythmically on a sandy "
        "shore, camera slowly pans right, cinematic quality, golden hour"
    ),
    # 4 – Food / café
    (
        "Close-up of a barista's hands pouring steamed milk into an espresso cup "
        "creating intricate latte art in the shape of a rosetta, warm cafe lighting, "
        "wooden countertop, steam rising gently, smooth pouring motion, bokeh"
    ),
    # 5 – Sci-fi / cyberpunk
    (
        "A futuristic cyberpunk city street at night with holographic billboards, "
        "flying vehicles and rain-slicked roads reflecting neon signs, a person "
        "in a glowing jacket walks toward the camera, volumetric lighting"
    ),
]

# ---------------------------------------------------------------------------
# Wan 2.2 recommended negative prompt (English translation from official)
# ---------------------------------------------------------------------------
WAN_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn "
    "hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused "
    "fingers, still picture, messy background, three legs, many people in the "
    "background, walking backwards"
)

# SDXL negative prompt
SDXL_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, distorted, ugly, deformed, "
    "watermark, text, signature, jpeg artifacts, out of frame, cropped"
)


def _sdxl_resolution(video_width: int, video_height: int) -> tuple:
    """Find the best SDXL resolution (~1 MP) matching the video aspect ratio."""
    aspect = video_width / video_height
    candidates = [
        (1024, 1024),   # 1:1
        (1152, 896),    # ~1.29:1
        (896, 1152),    # ~0.78:1
        (1216, 832),    # ~1.46:1
        (832, 1216),    # ~0.68:1
        (1344, 768),    # ~1.75:1
        (768, 1344),    # ~0.57:1
        (1536, 640),    # ~2.4:1
        (640, 1536),    # ~0.42:1
    ]
    best = min(candidates, key=lambda wh: abs(wh[0] / wh[1] - aspect))
    return best


def interpolate_frames(
    frames: list,
    multiplier: int = 2,
) -> list:
    """Insert interpolated frames between each pair for smoother playback.

    Uses alpha blending between adjacent frames. Simple but effective for
    increasing the perceived frame rate without extra model inference.

    Args:
        frames: List of PIL Image frames.
        multiplier: How many times to subdivide each inter-frame gap.
                    2 = insert 1 frame between each pair (doubles count).
    """
    if multiplier <= 1 or len(frames) < 2:
        return frames

    result: list = []
    for i in range(len(frames) - 1):
        result.append(frames[i])
        arr_a = np.array(frames[i], dtype=np.float32)
        arr_b = np.array(frames[i + 1], dtype=np.float32)
        for step in range(1, multiplier):
            alpha = step / multiplier
            blended = ((1.0 - alpha) * arr_a + alpha * arr_b).astype(np.uint8)
            result.append(Image.fromarray(blended))
    result.append(frames[-1])
    return result


class TikTokVideoGenerator:
    """Generate TikTok-style videos using SDXL + Wan 2.2.

    Two-stage pipeline:
      1. SDXL generates a high-quality starting frame from the text prompt
      2. Wan 2.2 TI2V-5B animates the frame into a 10-second video (I2V mode)
         — or generates a video from text alone (T2V mode)

    Models are loaded sequentially to minimise peak memory usage:
    the image model is unloaded before loading the video model.
    """

    DEFAULT_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    DEFAULT_VIDEO_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    def __init__(
        self,
        image_model: Optional[str] = None,
        video_model: Optional[str] = None,
        mode: str = "i2v",
    ):
        self.image_model_id = image_model or self.DEFAULT_IMAGE_MODEL
        self.video_model_id = video_model or self.DEFAULT_VIDEO_MODEL
        self.mode = mode  # "i2v" (image-to-video) or "t2v" (text-to-video)
        self.device = self._detect_device()
        self.image_pipeline = None
        self.video_pipeline = None
        self.gpu_monitor = GPUPowerMonitor()

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        if torch.backends.mps.is_available():
            print("[+] Apple Silicon GPU (MPS) detected")
            return "mps"
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[+] CUDA GPU detected: {name}")
            return "cuda"
        print("[!] No GPU found — falling back to CPU (this will be very slow)")
        return "cpu"

    def _clear_memory(self) -> None:
        """Force garbage collection and clear GPU cache."""
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Image model (SDXL)
    # ------------------------------------------------------------------

    def _load_image_model(self) -> None:
        """Load SDXL for image generation."""
        from diffusers import StableDiffusionXLPipeline

        print(f"\n[*] Loading image model: {self.image_model_id}")
        t0 = time.time()

        self.image_pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.image_model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        # Memory optimizations
        if self.device == "cuda":
            self.image_pipeline.enable_model_cpu_offload()
        else:
            self.image_pipeline.to(self.device)

        try:
            self.image_pipeline.enable_vae_slicing()
        except Exception:
            pass

        print(f"[+] Image model ready in {time.time() - t0:.1f} s")

    def _unload_image_model(self) -> None:
        """Unload SDXL to free memory for the video model."""
        if self.image_pipeline is not None:
            print("[*] Unloading image model to free memory...")
            del self.image_pipeline
            self.image_pipeline = None
            self._clear_memory()

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        video_width: int,
        video_height: int,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
    ) -> Image.Image:
        """Generate a high-quality image with SDXL, resized for video input.

        Generates at SDXL-optimal resolution (~1 MP) matching the video aspect
        ratio, then resizes to the exact video resolution.
        """
        if self.image_pipeline is None:
            self._load_image_model()

        # Generate at SDXL-optimal resolution
        sdxl_w, sdxl_h = _sdxl_resolution(video_width, video_height)
        generator = (
            torch.Generator("cpu").manual_seed(seed) if seed is not None else None
        )

        print(f"[*] Generating image at {sdxl_w}x{sdxl_h} (SDXL optimal)...")
        result = self.image_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=sdxl_w,
            height=sdxl_h,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
            generator=generator,
        )
        image = result.images[0]

        # Resize to video resolution
        if (image.width, image.height) != (video_width, video_height):
            print(
                f"[*] Resizing image: {image.width}x{image.height} "
                f"-> {video_width}x{video_height}"
            )
            image = image.resize((video_width, video_height), Image.LANCZOS)

        return image

    # ------------------------------------------------------------------
    # Video model (Wan 2.2)
    # ------------------------------------------------------------------

    def _load_video_model(self) -> None:
        """Load the Wan 2.2 video generation model."""
        t0 = time.time()

        if self.mode == "i2v":
            self._load_video_model_i2v()
        else:
            self._load_video_model_t2v()

        print(f"[+] Video model ready in {time.time() - t0:.1f} s")

    def _load_video_model_i2v(self) -> None:
        """Load Wan 2.2 TI2V-5B as WanImageToVideoPipeline for I2V.

        The TI2V-5B model supports image-to-video through VAE-based image
        conditioning (expand_timesteps mode). Components are loaded individually
        and assembled into a WanImageToVideoPipeline.
        """
        from diffusers import (
            AutoencoderKLWan,
            UniPCMultistepScheduler,
            WanImageToVideoPipeline,
            WanTransformer3DModel,
        )
        from transformers import AutoTokenizer, UMT5EncoderModel

        model_id = self.video_model_id
        print(f"\n[*] Loading Wan 2.2 I2V pipeline: {model_id}")

        print("[*] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, subfolder="tokenizer",
        )

        print("[*] Loading text encoder...")
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        )

        print("[*] Loading VAE (float32 for decoding quality)...")
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32,
        )

        print("[*] Loading transformer...")
        transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        )

        print("[*] Loading scheduler...")
        scheduler = UniPCMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler",
        )

        print("[*] Assembling I2V pipeline...")
        self.video_pipeline = WanImageToVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=None,       # Wan 2.2 uses VAE-only image conditioning
            image_processor=None,     # (no CLIP needed, unlike Wan 2.1 I2V)
            expand_timesteps=True,    # Required for TI2V-5B I2V mode
        )

        # Memory optimization: sequential offload reduces peak memory on MPS/CUDA
        if self.device == "cuda":
            self.video_pipeline.enable_model_cpu_offload()
        elif self.device == "mps":
            # Sequential offload keeps peak memory lower on Apple Silicon
            try:
                self.video_pipeline.enable_sequential_cpu_offload()
                print("[+] Using sequential CPU offload (MPS memory optimization)")
            except Exception:
                self.video_pipeline.to(self.device)
        else:
            self.video_pipeline.to(self.device)

        try:
            self.video_pipeline.enable_vae_slicing()
        except Exception:
            pass

    def _load_video_model_t2v(self) -> None:
        """Load Wan 2.2 TI2V-5B as WanPipeline for text-to-video."""
        from diffusers import AutoencoderKLWan, WanPipeline

        model_id = self.video_model_id
        print(f"\n[*] Loading Wan 2.2 T2V pipeline: {model_id}")

        print("[*] Loading VAE (float32 for decoding quality)...")
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32,
        )

        print("[*] Loading pipeline...")
        self.video_pipeline = WanPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch.bfloat16,
        )

        # Memory optimization
        if self.device == "cuda":
            self.video_pipeline.enable_model_cpu_offload()
        elif self.device == "mps":
            try:
                self.video_pipeline.enable_sequential_cpu_offload()
                print("[+] Using sequential CPU offload (MPS memory optimization)")
            except Exception:
                self.video_pipeline.to(self.device)
        else:
            self.video_pipeline.to(self.device)

        try:
            self.video_pipeline.enable_vae_slicing()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Full generation pipeline
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        preset: str = "tiktok_10s",
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
        interpolation_multiplier: int = 1,
        low_memory: bool = False,
    ) -> str:
        """Generate a video using the two-stage pipeline.

        Stage 1: SDXL generates a starting frame from the prompt.
        Stage 2: Wan 2.2 generates a video from the image (I2V) or prompt (T2V).

        Args:
            prompt: Text description of the video to create.
            negative_prompt: Things to avoid (defaults provided if omitted).
            preset: One of the keys in ``PRESETS``.
            num_inference_steps: Denoising steps for video generation.
            guidance_scale: How closely to follow the prompt.
            seed: Optional seed for reproducible results.
            output_path: Where to save the .mp4 (auto-generated if omitted).
            interpolation_multiplier: Frame interpolation factor
                                      (1 = off, 2 = double fps).

        Returns:
            Path to the saved video file.
        """
        from diffusers.utils import export_to_video

        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}"
            )

        cfg = PRESETS[preset]
        width = cfg["width"]
        height = cfg["height"]
        num_frames = cfg["num_frames"]
        base_fps = cfg["fps"]

        # On MPS (Apple Silicon), 10s (161 frames) often causes OOM; cap at 5s (81 frames)
        if (self.device == "mps" or low_memory) and num_frames > 81:
            print(
                "[*] MPS/low-memory: capping at 81 frames (~5 s) to avoid OOM. "
                "Use --low-memory on other devices for the same cap."
            )
            num_frames = 81

        # Calculate effective output with interpolation
        if interpolation_multiplier > 1:
            effective_frames = (num_frames - 1) * interpolation_multiplier + 1
            effective_fps = base_fps * interpolation_multiplier
        else:
            effective_frames = num_frames
            effective_fps = base_fps

        duration_s = effective_frames / effective_fps

        if negative_prompt is None:
            negative_prompt = WAN_NEGATIVE_PROMPT

        # Generator for reproducibility
        generator = (
            torch.Generator("cpu").manual_seed(seed) if seed is not None else None
        )

        # Pretty-print settings
        mode_label = (
            "Image -> Video (I2V)" if self.mode == "i2v"
            else "Text -> Video (T2V)"
        )
        print("\n" + "=" * 62)
        print("  TikTok Video Generation (SDXL + Wan 2.2)")
        print("=" * 62)
        print(f"  Mode           : {mode_label}")
        print(f"  Preset         : {preset} — {cfg['description']}")
        print(f"  Resolution     : {width} x {height}")
        print(f"  Gen frames     : {num_frames} @ {base_fps} fps")
        if interpolation_multiplier > 1:
            print(
                f"  Output frames  : {effective_frames} @ {effective_fps} fps "
                f"({interpolation_multiplier}x interpolation)"
            )
        print(f"  Duration       : ~{duration_s:.1f} s")
        print(f"  Video steps    : {num_inference_steps}")
        print(f"  Guidance       : {guidance_scale}")
        if seed is not None:
            print(f"  Seed           : {seed}")
        print(
            f"  Prompt         : {prompt[:76]}{'...' if len(prompt) > 76 else ''}"
        )
        print("=" * 62)

        # Determine output paths
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"tiktok_{ts}"
        else:
            base_name = os.path.splitext(output_path)[0]

        # ── Stage 1: Image generation ────────────────────────────
        print("\n" + "-" * 50)
        print("  Stage 1: Generating Starting Image (SDXL)")
        print("-" * 50)

        self._load_image_model()
        self.gpu_monitor.start_monitoring(interval=1.0)
        t_img_start = time.time()

        image = self.generate_image(
            prompt=prompt,
            negative_prompt=SDXL_NEGATIVE_PROMPT,
            video_width=width,
            video_height=height,
            seed=seed,
        )

        t_img_elapsed = time.time() - t_img_start
        self.gpu_monitor.stop_monitoring()

        # Save the generated image
        image_path = f"{base_name}_frame.png"
        image.save(image_path)
        print(f"\n[+] Starting image saved: {image_path} ({t_img_elapsed:.1f} s)")

        stats = self.gpu_monitor.get_statistics()
        if "gpu_power_avg_watts" in stats:
            print(f"    Avg GPU power: {stats['gpu_power_avg_watts']:.2f} W")

        # Free image model memory before loading video model
        self._unload_image_model()

        # ── Stage 2: Video generation ────────────────────────────
        print("\n" + "-" * 50)
        print("  Stage 2: Generating Video (Wan 2.2)")
        print("-" * 50)

        self._load_video_model()
        self.gpu_monitor.start_monitoring(interval=1.0)
        t_vid_start = time.time()

        try:
            if self.mode == "i2v":
                result = self.video_pipeline(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            else:
                result = self.video_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

            video_frames = result.frames[0]
        finally:
            self.gpu_monitor.stop_monitoring()

        t_vid_elapsed = time.time() - t_vid_start

        # Report GPU stats for video generation
        stats = self.gpu_monitor.get_statistics()
        print(f"\n[+] Video generation finished in {t_vid_elapsed:.1f} s")
        print(f"    Monitoring samples: {stats.get('total_samples', 0)}")
        if "gpu_power_avg_watts" in stats:
            print(f"    Avg GPU power     : {stats['gpu_power_avg_watts']:.2f} W")
            print(f"    Peak GPU power    : {stats['gpu_power_max_watts']:.2f} W")

        # ── Post-processing: optional frame interpolation ────────
        if interpolation_multiplier > 1:
            print(
                f"[*] Interpolating frames ({len(video_frames)} -> "
                f"{(len(video_frames) - 1) * interpolation_multiplier + 1})..."
            )
            t_interp = time.time()
            video_frames = interpolate_frames(
                video_frames, interpolation_multiplier,
            )
            print(f"    Done in {time.time() - t_interp:.1f} s")

        # Save video
        video_path = output_path or f"{base_name}.mp4"
        export_to_video(video_frames, video_path, fps=effective_fps)

        total_elapsed = t_img_elapsed + t_vid_elapsed
        print(f"\n[+] Video saved  : {video_path}")
        print(f"    Image        : {image_path}")
        print(f"    Duration     : ~{duration_s:.1f} s")
        print(f"    FPS          : {effective_fps}")
        print(f"    Frames       : {len(video_frames)}")
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        print(f"    File size    : {file_size:.1f} MB")
        print(f"    Total time   : {total_elapsed:.1f} s")

        return video_path


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TikTok-style videos using SDXL + Wan 2.2 "
                    "on a local GPU (Apple Silicon MPS / CUDA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python video_generator.py "A puppy surfing on a tiny wave"\n'
            "  python video_generator.py --example 3 --preset tiktok_10s\n"
            "  python video_generator.py --mode t2v --example 5\n"
            "  python video_generator.py --list-presets\n"
            "  python video_generator.py --list-examples\n"
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Text prompt describing the video to generate",
    )
    parser.add_argument(
        "--mode",
        choices=["i2v", "t2v"],
        default="i2v",
        help="Generation mode: i2v (image-to-video, default) or t2v (text-to-video)",
    )
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        default="tiktok_10s",
        help="Video size / duration preset (default: tiktok_10s)",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=50,
        help="Video denoising steps — lower is faster, higher is better quality "
             "(default: 50)",
    )
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=5.0,
        help="Guidance scale — higher = more prompt adherence (default: 5.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output .mp4 path (default: tiktok_<timestamp>.mp4)",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default=None,
        help=f"Image gen model (default: {TikTokVideoGenerator.DEFAULT_IMAGE_MODEL})",
    )
    parser.add_argument(
        "--video-model",
        type=str,
        default=None,
        help=f"Video gen model (default: {TikTokVideoGenerator.DEFAULT_VIDEO_MODEL})",
    )
    parser.add_argument(
        "--interpolation", "-i",
        type=int,
        default=1,
        metavar="N",
        help="Frame interpolation multiplier (default: 1 = off, 2 = double fps)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Cap at 81 frames (~5 s) to reduce memory use (recommended on Apple Silicon)",
    )
    parser.add_argument(
        "--example", "-e",
        type=int,
        default=None,
        metavar="N",
        help=f"Use built-in example prompt (0-{len(EXAMPLE_PROMPTS) - 1})",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Show available presets and exit",
    )
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="Show built-in example prompts and exit",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ---- Info-only flags ----
    if args.list_presets:
        print("\nAvailable presets:\n")
        for name, cfg in PRESETS.items():
            dur = cfg["num_frames"] / cfg["fps"]
            print(
                f"  {name:18s}  {cfg['width']:>4d}x{cfg['height']:<4d}  "
                f"{cfg['num_frames']:>3d} frames @ {cfg['fps']} fps  "
                f"~{dur:.0f} s  — {cfg['description']}"
            )
        print()
        return

    if args.list_examples:
        print("\nExample prompts:\n")
        for i, p in enumerate(EXAMPLE_PROMPTS):
            print(f"  [{i}] {p[:90]}{'...' if len(p) > 90 else ''}")
        print()
        return

    # ---- Resolve prompt ----
    prompt = args.prompt
    if prompt is None and args.example is not None:
        if 0 <= args.example < len(EXAMPLE_PROMPTS):
            prompt = EXAMPLE_PROMPTS[args.example]
        else:
            parser.error(
                f"--example must be 0-{len(EXAMPLE_PROMPTS) - 1}, "
                f"got {args.example}"
            )
    if prompt is None:
        print("[*] No prompt given — using example prompt [0].\n")
        prompt = EXAMPLE_PROMPTS[0]

    # ---- Generate ----
    gen = TikTokVideoGenerator(
        image_model=args.image_model,
        video_model=args.video_model,
        mode=args.mode,
    )

    output = gen.generate(
        prompt=prompt,
        preset=args.preset,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        output_path=args.output,
        interpolation_multiplier=args.interpolation,
        low_memory=args.low_memory,
    )

    print(f"\nDone! Your TikTok-style video is ready: {output}")


if __name__ == "__main__":
    main()
