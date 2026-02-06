"""
TikTok-Style Video Generator — Generate short vertical videos from text prompts.

Uses AnimateDiff (via HuggingFace Diffusers) for high-quality text-to-video
generation on Apple Silicon GPU (MPS backend).

Improvements over LTX-Video baseline:
  - AnimateDiff with Stable Diffusion 1.5 backbone (much higher visual quality)
  - Realistic_Vision_V5.1 base model (top-tier photorealistic SD 1.5 finetune)
  - sd-vae-ft-mse VAE for sharper frame decoding
  - FreeInit for improved temporal consistency (no training required)
  - FreeNoise for generating longer videos (32–96+ frames via sliding window)
  - Motion LoRA support for camera movements (zoom, pan, tilt)
  - Post-processing frame interpolation for smoother playback
  - Automatic prompt enhancement with quality tags

Model stack (~4 GB in fp16):
  - Base: SG161222/Realistic_Vision_V5.1_noVAE
  - Motion: guoyww/animatediff-motion-adapter-v1-5-2
  - VAE:  stabilityai/sd-vae-ft-mse
  - Scheduler: DDIM (linear beta, no clip_sample)

Usage:
  python video_generator.py "A cat skateboarding through a neon city"
  python video_generator.py --preset tiktok_medium --example 2
  python video_generator.py --preset tiktok_long --example 3 --motion zoom-out
  python video_generator.py --list-presets
  python video_generator.py --list-examples
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional, List

import numpy as np
import torch
from PIL import Image

from gpu_monitor import GPUPowerMonitor


# ---------------------------------------------------------------------------
# TikTok / short-form video presets
# AnimateDiff native resolution is 512x512 (SD 1.5 based).
# Portrait 512x768 and landscape 768x512 work well.
# Frame counts are designed for FreeNoise sliding-window generation.
# ---------------------------------------------------------------------------
PRESETS = {
    "tiktok_short": {
        "width": 512,
        "height": 768,
        "num_frames": 32,        # ~4 s @ 8 fps  (-> 8 s @ 16 fps with interpolation)
        "fps": 8,
        "description": "Short TikTok clip (portrait, ~4 s)",
    },
    "tiktok_medium": {
        "width": 512,
        "height": 768,
        "num_frames": 48,        # ~6 s @ 8 fps
        "fps": 8,
        "description": "Medium TikTok clip (portrait, ~6 s)",
    },
    "tiktok_long": {
        "width": 512,
        "height": 768,
        "num_frames": 64,        # ~8 s @ 8 fps
        "fps": 8,
        "description": "Longer TikTok clip (portrait, ~8 s)",
    },
    "square": {
        "width": 512,
        "height": 512,
        "num_frames": 32,        # ~4 s @ 8 fps
        "fps": 8,
        "description": "Square video (1:1, ~4 s)",
    },
    "landscape": {
        "width": 768,
        "height": 512,
        "num_frames": 32,        # ~4 s @ 8 fps
        "fps": 8,
        "description": "Landscape video (3:2, ~4 s)",
    },
}

# ---------------------------------------------------------------------------
# Motion LoRAs — camera movement presets (work with motion-adapter-v1-5-2)
# ---------------------------------------------------------------------------
MOTION_LORAS = {
    "zoom-in": "guoyww/animatediff-motion-lora-zoom-in",
    "zoom-out": "guoyww/animatediff-motion-lora-zoom-out",
    "pan-left": "guoyww/animatediff-motion-lora-pan-left",
    "pan-right": "guoyww/animatediff-motion-lora-pan-right",
    "tilt-up": "guoyww/animatediff-motion-lora-tilt-up",
    "tilt-down": "guoyww/animatediff-motion-lora-tilt-down",
    "roll-clockwise": "guoyww/animatediff-motion-lora-rolling-clockwise",
    "roll-anticlockwise": "guoyww/animatediff-motion-lora-rolling-anticlockwise",
}

# ---------------------------------------------------------------------------
# Example prompts — TikTok-style content ideas (with quality tags for SD 1.5)
# ---------------------------------------------------------------------------
EXAMPLE_PROMPTS = [
    # 0 – Dance / neon vibe
    (
        "masterpiece, best quality, highly detailed, "
        "a person dancing energetically in a neon-lit room with colorful purple "
        "and blue lights reflecting off the walls, dynamic camera movement follows "
        "the dancer, vibrant electric atmosphere, cinematic lighting, "
        "photorealistic, 8k uhd, dslr"
    ),
    # 1 – Cute animal
    (
        "masterpiece, best quality, highly detailed, "
        "a golden retriever puppy wearing tiny sunglasses sitting on a surfboard "
        "riding a small wave at the beach, close-up shot, the puppy looks excited, "
        "bright sunny day, turquoise ocean water splashing gently, "
        "photorealistic, sharp focus, cinematic"
    ),
    # 2 – Satisfying / ASMR
    (
        "masterpiece, best quality, highly detailed, "
        "satisfying top-down footage of colorful acrylic paint being poured and "
        "swirled in slow motion on a white canvas, vibrant pink teal and gold "
        "colors merge into mesmerizing patterns, soft studio lighting, "
        "macro photography, sharp focus"
    ),
    # 3 – Nature timelapse
    (
        "masterpiece, best quality, highly detailed, "
        "a breathtaking timelapse of a sunset over the ocean, golden and orange "
        "clouds move across the sky while waves crash rhythmically on a "
        "sandy shore, camera slowly pans right, cinematic quality, "
        "landscape photography, golden hour, 8k uhd"
    ),
    # 4 – Food / café
    (
        "masterpiece, best quality, highly detailed, "
        "close-up of a barista's hands pouring steamed milk into an espresso cup "
        "creating intricate latte art in the shape of a rosetta, warm café lighting, "
        "wooden countertop, steam rising gently, smooth pouring motion, "
        "food photography, bokeh background"
    ),
    # 5 – Sci-fi / cyberpunk
    (
        "masterpiece, best quality, highly detailed, "
        "a futuristic cyberpunk city street at night with holographic billboards "
        "flying vehicles and rain-slicked roads reflecting neon signs, a person "
        "in a glowing jacket walks toward the camera, cinematic depth of field, "
        "volumetric lighting, photorealistic, 8k uhd"
    ),
]


def interpolate_frames(
    frames: List[Image.Image],
    multiplier: int = 2,
) -> List[Image.Image]:
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

    result: List[Image.Image] = []
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
    """Generate TikTok-style vertical videos from text prompts using local GPU.

    Wraps the AnimateDiff pipeline (Stable Diffusion 1.5 + Motion Adapter)
    with quality enhancements tuned for Apple Silicon (MPS) and optional
    GPU power monitoring.
    """

    DEFAULT_BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
    DEFAULT_MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
    DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"

    def __init__(
        self,
        base_model: Optional[str] = None,
        motion_adapter: Optional[str] = None,
        vae_model: Optional[str] = None,
    ):
        self.base_model = base_model or self.DEFAULT_BASE_MODEL
        self.motion_adapter_id = motion_adapter or self.DEFAULT_MOTION_ADAPTER
        self.vae_model_id = vae_model or self.DEFAULT_VAE
        self.device = self._detect_device()
        self.dtype = self._optimal_dtype()
        self.pipeline = None
        self.gpu_monitor = GPUPowerMonitor()

    # ------------------------------------------------------------------
    # Device / dtype helpers
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

    def _optimal_dtype(self) -> torch.dtype:
        """Pick the best dtype for the active device."""
        if self.device == "cuda":
            return torch.float16
        if self.device == "mps":
            return torch.float16
        return torch.float32

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download (if needed) and load the AnimateDiff pipeline."""
        from diffusers import AnimateDiffPipeline, DDIMScheduler, AutoencoderKL
        from diffusers.models import MotionAdapter

        print(f"\n[*] Loading AnimateDiff pipeline")
        print(f"    Base model      : {self.base_model}")
        print(f"    Motion adapter  : {self.motion_adapter_id}")
        print(f"    VAE             : {self.vae_model_id}")
        print(f"    Device          : {self.device}")
        print(f"    Dtype           : {self.dtype}")

        t0 = time.time()

        # 1. Load motion adapter
        print("[*] Loading motion adapter...")
        adapter = MotionAdapter.from_pretrained(
            self.motion_adapter_id,
            torch_dtype=self.dtype,
        )

        # 2. Load VAE (sd-vae-ft-mse produces sharper frames than default)
        print("[*] Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            self.vae_model_id,
            torch_dtype=self.dtype,
        )

        # 3. Load main pipeline with motion adapter + VAE
        print("[*] Loading base model + assembling pipeline...")
        self.pipeline = AnimateDiffPipeline.from_pretrained(
            self.base_model,
            motion_adapter=adapter,
            vae=vae,
            torch_dtype=self.dtype,
        )

        # 4. Configure scheduler — DDIM with linear beta (recommended for AnimateDiff)
        scheduler = DDIMScheduler.from_pretrained(
            self.base_model,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        self.pipeline.scheduler = scheduler

        # 5. Move to device
        self.pipeline.to(self.device)

        # 6. Memory optimizations
        self.pipeline.enable_vae_slicing()

        if self.device == "mps":
            # MPS does NOT support Flash Attention or memory-efficient attention.
            # PyTorch SDPA falls back to the "math" backend which materializes the
            # full (batch*heads, seq, seq) attention matrix — for video generation
            # (batch = num_frames * 2 for CFG) this easily exceeds the maximum
            # Metal buffer size.  Replace with SlicedAttnProcessor which processes
            # attention in small chunks that fit in memory.
            try:
                from diffusers.models.attention_processor import SlicedAttnProcessor

                self.pipeline.unet.set_attn_processor(
                    SlicedAttnProcessor(slice_size=4)
                )
                print("[+] Using sliced attention (MPS memory optimization)")
            except ImportError:
                # Fallback for older diffusers — set aggressive attention slicing
                self.pipeline.enable_attention_slicing(1)
                print("[+] Using attention slicing fallback (MPS)")
        else:
            self.pipeline.enable_attention_slicing()

        # VAE tiling reduces peak memory for larger resolutions
        try:
            self.pipeline.enable_vae_tiling()
        except Exception:
            pass

        # Forward chunking saves memory by processing feed-forward layers in chunks
        try:
            self.pipeline.unet.enable_forward_chunking(chunk_size=4)
        except Exception:
            pass

        elapsed = time.time() - t0
        print(f"[+] Pipeline ready in {elapsed:.1f} s\n")

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    @staticmethod
    def _enhance_prompt(prompt: str) -> str:
        """Prepend quality tags if user prompt doesn't already include them."""
        quality_tags = [
            "masterpiece", "best quality", "highly detailed",
        ]
        lower = prompt.lower()
        # Don't double-add quality tags
        if any(tag in lower for tag in quality_tags):
            return prompt
        prefix = "masterpiece, best quality, highly detailed, "
        return prefix + prompt

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        preset: str = "tiktok_short",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
        motion_lora: Optional[str] = None,
        use_free_init: bool = False,
        use_free_noise: bool = True,
        interpolation_multiplier: int = 2,
        enhance_prompt: bool = True,
    ) -> str:
        """Generate a short video and save it to disk.

        Args:
            prompt: Text description of the video to create.
            negative_prompt: Things to avoid (default: common quality issues).
            preset: One of the keys in ``PRESETS``.
            num_inference_steps: Denoising steps (more = higher quality, slower).
            guidance_scale: How closely to follow the prompt (higher = stricter).
            seed: Optional seed for reproducible results.
            output_path: Where to save the .mp4 (auto-generated if omitted).
            motion_lora: Camera motion preset (e.g. "zoom-out", "pan-left").
            use_free_init: Enable FreeInit for better temporal consistency.
            use_free_noise: Enable FreeNoise for longer videos (>16 frames).
            interpolation_multiplier: Frame interpolation factor (2 = double fps).
            enhance_prompt: Auto-add quality tags to the prompt.

        Returns:
            The path to the saved video file.
        """
        from diffusers.utils import export_to_video

        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded — call load_model() first.")

        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}"
            )

        cfg = PRESETS[preset]
        width = cfg["width"]
        height = cfg["height"]
        num_frames = cfg["num_frames"]
        base_fps = cfg["fps"]

        # Enhance prompt with quality tags
        if enhance_prompt:
            prompt = self._enhance_prompt(prompt)

        if negative_prompt is None:
            negative_prompt = (
                "worst quality, low quality, bad quality, lowres, blurry, "
                "jittery, distorted, ugly, deformed, disfigured, watermark, "
                "text, signature, jpeg artifacts, out of frame, cropped, "
                "bad anatomy, bad proportions, duplicate, error"
            )

        # Generator for reproducibility (always on CPU — works with any device)
        generator = (
            torch.Generator("cpu").manual_seed(seed) if seed is not None else None
        )

        # --- Enable Motion LoRA (camera movement) ---
        if motion_lora:
            lora_key = motion_lora.lower().replace(" ", "-")
            if lora_key in MOTION_LORAS:
                try:
                    print(f"[*] Loading motion LoRA: {lora_key}")
                    self.pipeline.load_lora_weights(
                        MOTION_LORAS[lora_key],
                        adapter_name="motion_lora",
                    )
                    self.pipeline.set_adapters(["motion_lora"], [0.8])
                except Exception as e:
                    print(f"[!] Could not load motion LoRA: {e}")
            else:
                print(
                    f"[!] Unknown motion LoRA '{motion_lora}'. "
                    f"Available: {list(MOTION_LORAS.keys())}"
                )

        # --- Enable FreeNoise for longer videos ---
        free_noise_enabled = False
        if use_free_noise and num_frames > 16:
            try:
                self.pipeline.enable_free_noise(
                    context_length=16,
                    context_stride=4,
                )
                # Split inference reduces memory for FreeNoise by chunking across
                # spatial and temporal batch dimensions.
                if self.device == "mps":
                    try:
                        self.pipeline.enable_free_noise_split_inference()
                    except Exception:
                        pass
                free_noise_enabled = True
                print("[+] FreeNoise enabled (longer video via sliding window)")
            except Exception as e:
                print(f"[!] FreeNoise not available: {e}")
                # Fall back to max 16 frames without FreeNoise
                num_frames = min(num_frames, 16)
                print(f"    Capped to {num_frames} frames")

        # --- Enable FreeInit for temporal consistency ---
        free_init_enabled = False
        if use_free_init:
            try:
                self.pipeline.enable_free_init(
                    method="butterworth",
                    use_fast_sampling=True,
                )
                free_init_enabled = True
                print("[+] FreeInit enabled (better temporal consistency)")
            except Exception as e:
                print(f"[!] FreeInit not available: {e}")

        # Calculate effective output
        effective_frames = num_frames
        effective_fps = base_fps
        if interpolation_multiplier > 1:
            effective_frames = (num_frames - 1) * interpolation_multiplier + 1
            effective_fps = base_fps * interpolation_multiplier

        duration_s = effective_frames / effective_fps

        # Pretty-print settings
        print("=" * 62)
        print("  TikTok Video Generation (AnimateDiff)")
        print("=" * 62)
        print(f"  Preset         : {preset} — {cfg['description']}")
        print(f"  Resolution     : {width} x {height}")
        print(f"  Gen frames     : {num_frames} @ {base_fps} fps")
        if interpolation_multiplier > 1:
            print(
                f"  Output frames  : {effective_frames} @ {effective_fps} fps "
                f"({interpolation_multiplier}x interpolation)"
            )
        print(f"  Duration       : ~{duration_s:.1f} s")
        print(f"  Steps          : {num_inference_steps}")
        print(f"  Guidance       : {guidance_scale}")
        print(f"  FreeNoise      : {'ON' if free_noise_enabled else 'OFF'}")
        print(f"  FreeInit       : {'ON' if free_init_enabled else 'OFF'}")
        if motion_lora:
            print(f"  Motion LoRA    : {motion_lora}")
        if seed is not None:
            print(f"  Seed           : {seed}")
        print(f"  Prompt         : {prompt[:76]}{'…' if len(prompt) > 76 else ''}")
        print("=" * 62)

        # Start GPU monitoring in background
        self.gpu_monitor.start_monitoring(interval=1.0)

        t0 = time.time()
        try:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            video_frames = result.frames[0]
        finally:
            self.gpu_monitor.stop_monitoring()

            # Disable FreeInit / FreeNoise after generation to avoid side effects
            if free_init_enabled:
                try:
                    self.pipeline.disable_free_init()
                except Exception:
                    pass
            if free_noise_enabled:
                try:
                    self.pipeline.disable_free_noise()
                except Exception:
                    pass

            # Unload Motion LoRA if loaded
            if motion_lora and motion_lora.lower().replace(" ", "-") in MOTION_LORAS:
                try:
                    self.pipeline.unload_lora_weights()
                except Exception:
                    pass

        gen_elapsed = time.time() - t0

        # Report GPU statistics
        stats = self.gpu_monitor.get_statistics()
        print(f"\n[+] Generation finished in {gen_elapsed:.1f} s")
        print(f"    Monitoring samples : {stats.get('total_samples', 0)}")
        if "gpu_power_avg_watts" in stats:
            print(f"    Avg GPU power      : {stats['gpu_power_avg_watts']:.2f} W")
            print(f"    Peak GPU power     : {stats['gpu_power_max_watts']:.2f} W")

        # --- Post-processing: frame interpolation ---
        if interpolation_multiplier > 1:
            print(
                f"[*] Interpolating frames ({len(video_frames)} -> "
                f"{(len(video_frames) - 1) * interpolation_multiplier + 1})..."
            )
            t_interp = time.time()
            video_frames = interpolate_frames(video_frames, interpolation_multiplier)
            print(f"    Done in {time.time() - t_interp:.1f} s")

        # Determine output path
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"tiktok_{ts}.mp4"

        export_to_video(video_frames, output_path, fps=effective_fps)

        print(f"\n[+] Video saved  : {output_path}")
        print(f"    Duration     : ~{duration_s:.1f} s")
        print(f"    FPS          : {effective_fps}")
        print(f"    Frames       : {len(video_frames)}")
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    File size    : {file_size:.1f} MB")

        return output_path


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TikTok-style vertical videos from text prompts "
                    "using AnimateDiff on a local GPU (Apple Silicon MPS / CUDA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python video_generator.py "A puppy surfing on a tiny wave"\n'
            "  python video_generator.py --example 3 --preset tiktok_medium\n"
            "  python video_generator.py --example 5 --motion zoom-out\n"
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
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        default="tiktok_short",
        help="Video size / duration preset (default: tiktok_short)",
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=25,
        help="Denoising steps — lower is faster, higher is better quality (default: 25)",
    )
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=7.5,
        help="Guidance scale — higher = more prompt adherence (default: 7.5)",
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
        "--model",
        type=str,
        default=None,
        help="HuggingFace base model ID (default: SG161222/Realistic_Vision_V5.1_noVAE)",
    )
    parser.add_argument(
        "--motion", "-m",
        type=str,
        default=None,
        metavar="MOTION",
        help=f"Camera motion LoRA: {', '.join(MOTION_LORAS.keys())}",
    )
    parser.add_argument(
        "--free-init",
        action="store_true",
        help="Enable FreeInit for better temporal consistency (slower, ~3x compute)",
    )
    parser.add_argument(
        "--no-free-noise",
        action="store_true",
        help="Disable FreeNoise (limits to 16 frames max)",
    )
    parser.add_argument(
        "--no-interpolation",
        action="store_true",
        help="Disable post-processing frame interpolation",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable automatic prompt enhancement with quality tags",
    )
    parser.add_argument(
        "--interpolation", "-i",
        type=int,
        default=2,
        metavar="N",
        help="Frame interpolation multiplier (default: 2, set 1 to disable)",
    )
    parser.add_argument(
        "--example", "-e",
        type=int,
        default=None,
        metavar="N",
        help=f"Use built-in example prompt (0–{len(EXAMPLE_PROMPTS) - 1})",
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
    parser.add_argument(
        "--list-motions",
        action="store_true",
        help="Show available motion LoRAs and exit",
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
                f"{cfg['num_frames']:>3d} frames  ~{dur:.0f} s  — {cfg['description']}"
            )
        print()
        return

    if args.list_examples:
        print("\nExample prompts:\n")
        for i, p in enumerate(EXAMPLE_PROMPTS):
            print(f"  [{i}] {p[:90]}{'…' if len(p) > 90 else ''}")
        print()
        return

    if args.list_motions:
        print("\nAvailable motion LoRAs:\n")
        for name, repo in MOTION_LORAS.items():
            print(f"  {name:22s}  {repo}")
        print()
        return

    # ---- Resolve prompt ----
    prompt = args.prompt
    if prompt is None and args.example is not None:
        if 0 <= args.example < len(EXAMPLE_PROMPTS):
            prompt = EXAMPLE_PROMPTS[args.example]
        else:
            parser.error(
                f"--example must be 0–{len(EXAMPLE_PROMPTS) - 1}, got {args.example}"
            )
    if prompt is None:
        print("[*] No prompt given — using example prompt [0].\n")
        prompt = EXAMPLE_PROMPTS[0]

    # ---- Interpolation multiplier ----
    interp = 1 if args.no_interpolation else args.interpolation

    # ---- Generate ----
    gen = TikTokVideoGenerator(base_model=args.model)
    gen.load_model()

    output = gen.generate(
        prompt=prompt,
        preset=args.preset,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        output_path=args.output,
        motion_lora=args.motion,
        use_free_init=args.free_init,
        use_free_noise=not args.no_free_noise,
        interpolation_multiplier=interp,
        enhance_prompt=not args.no_enhance,
    )

    print(f"\nDone! Your TikTok-style video is ready: {output}")


if __name__ == "__main__":
    main()
