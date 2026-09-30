"""Backend registry; each adapter owns model loading and artifact production."""
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from .video import padded_frames


class Backend(Protocol):
    label: str

    def reconstruct(self, frames: np.ndarray, directory: Path, log) -> tuple: ...


class FluxVAE:
    label = "FLUX KinoVAE · encode / decode"

    def __init__(self, weights: Path):
        self.weights = weights
        self.model = None

    def reconstruct(self, frames, directory, log):
        import torch
        from flux_action.models.video_vae import load_video_vae
        from safetensors.torch import save_file

        if not self.weights.is_file():
            raise FileNotFoundError(f"VAE ağırlığı bulunamadı: {self.weights}")
        if self.model is None:
            log("model", f"KinoVAE yükleniyor: {self.weights}")
            self.model = load_video_vae(str(self.weights), device="cuda", compile_model=False)
            log("model", f"GPU: {torch.cuda.get_device_name(0)}; dtype=bfloat16; decoder_max_t=8")
        count = len(frames)
        padding = padded_frames(count) - count
        pixels = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        pixels = pixels.to(device="cuda", dtype=torch.bfloat16).div_(127.5).sub_(1)
        if padding:
            pixels = torch.cat([pixels, pixels[:, :, -1:].expand(-1, -1, padding, -1, -1)], dim=2)
        torch.cuda.reset_peak_memory_stats()
        log("encode", f"RGB [-1,1] → normalize latent; input={list(pixels.shape)}; temporal padding={padding}")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            latents = self.model.encode(pixels)
        torch.cuda.synchronize()
        encode_seconds = time.perf_counter() - started
        del pixels
        log("encode", f"Encode tamamlandı ({encode_seconds:.3f}s); latent={list(latents.shape)}")
        save_file({"latents": latents.contiguous().cpu()}, str(directory / "latents.safetensors"), metadata={"backend": "flux_vae", "normalization": "KinoVAE normalized latents", "original_frames": str(count), "padded_frames": str(count + padding)})
        log("decode", "Normalize latent → RGB; tam-sekans decode, 8-kare aktivasyon penceresi")
        started = time.perf_counter()
        with torch.inference_mode():
            decoded = self.model.decode(latents)
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - started
        result = decoded[0, :, :count].permute(1, 2, 3, 0).float().add_(1).mul_(127.5).clamp_(0, 255).round_().to(torch.uint8).cpu().numpy()
        metrics = {"encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
                   "latent_shape": list(latents.shape), "padded_frames": count + padding,
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                   "dtype": "bfloat16", "weights": str(self.weights), "decoder_max_t": 8}
        del decoded, latents
        torch.cuda.empty_cache()
        log("decode", f"Decode tamamlandı ({decode_seconds:.3f}s); {count} kare")
        return result, metrics


def registry(weights: Path) -> dict[str, Backend]:
    return {"flux_vae": FluxVAE(weights)}
