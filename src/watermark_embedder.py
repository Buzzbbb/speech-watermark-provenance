"""不可感知嵌入模块 – WatermarkEmbedder

Implements two complementary watermark embedding algorithms:

1. **Spread-Spectrum (SS)** – adds a scaled, keyed PN sequence to each frame
   in the time domain.  The scale factor adapts to the local RMS energy so
   that the watermark remains perceptually masked.

2. **DCT-domain** – embeds bits by modifying pairs of mid-frequency DCT
   coefficients, similar to the Bender et al. "Echo Hiding" family of
   methods.  More robust to low-pass filtering.

Both methods embed the bit-stream produced by :class:`WatermarkEncoder`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.fft import dct, idct

from .watermark_encoder import WatermarkEncoder


class WatermarkEmbedder:
    """Embed a watermark bit-stream imperceptibly into audio.

    Parameters
    ----------
    encoder : WatermarkEncoder
        Encoder used to generate the bit-stream and PN sequences.
    frame_length : int
        Number of samples per embedding frame.
    alpha : float
        Base watermark strength.  The actual per-frame strength is scaled by
        the local RMS energy (adaptive masking).
    method : {'ss', 'dct'}
        Embedding algorithm.  'ss' = spread-spectrum; 'dct' = DCT-domain.
    dct_coef_start : int
        First DCT coefficient index used for DCT-domain embedding.
    dct_coef_count : int
        Number of consecutive DCT coefficients modified per bit.
    """

    METHODS = ("ss", "dct")

    def __init__(
        self,
        encoder: Optional[WatermarkEncoder] = None,
        frame_length: int = 256,
        alpha: float = 0.05,
        method: str = "ss",
        dct_coef_start: int = 20,
        dct_coef_count: int = 10,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.encoder = encoder or WatermarkEncoder()
        self.frame_length = frame_length
        self.alpha = alpha
        self.method = method
        self.dct_coef_start = dct_coef_start
        self.dct_coef_count = dct_coef_count

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def embed(self, audio: np.ndarray, message: str) -> np.ndarray:
        """Embed *message* into *audio* and return the watermarked signal.

        Parameters
        ----------
        audio : np.ndarray, shape (N,)
            Input mono audio (float32).
        message : str
            Text message to embed.

        Returns
        -------
        watermarked : np.ndarray, shape (N,), dtype float32

        Raises
        ------
        ValueError
            When the audio is too short to embed the full message.
        """
        bit_stream = self.encoder.encode(message)
        n_frames = len(audio) // self.frame_length
        if n_frames < len(bit_stream):
            raise ValueError(
                f"Audio too short: need {len(bit_stream)} frames "
                f"({len(bit_stream) * self.frame_length} samples), "
                f"got {n_frames} frames ({len(audio)} samples)."
            )

        if self.method == "ss":
            return self._embed_ss(audio, bit_stream)
        return self._embed_dct(audio, bit_stream)

    # ------------------------------------------------------------------
    # Spread-Spectrum embedding
    # ------------------------------------------------------------------

    def _embed_ss(self, audio: np.ndarray, bit_stream: np.ndarray) -> np.ndarray:
        """Time-domain spread-spectrum embedding."""
        out = audio.astype(np.float32).copy()
        L = self.frame_length

        for i, bit in enumerate(bit_stream):
            start = i * L
            frame = out[start : start + L]
            pn = self.encoder.generate_pn(i, L)

            # Adaptive scale: proportional to local RMS
            rms = float(np.sqrt(np.mean(frame ** 2))) + 1e-9
            scale = self.alpha * rms

            # Bipolar bit: 0 → -1, 1 → +1
            polarity = 2.0 * float(bit) - 1.0
            out[start : start + L] += scale * polarity * pn

        # Hard-clip to [-1, 1]
        return np.clip(out, -1.0, 1.0)

    # ------------------------------------------------------------------
    # DCT-domain embedding
    # ------------------------------------------------------------------

    def _embed_dct(self, audio: np.ndarray, bit_stream: np.ndarray) -> np.ndarray:
        """DCT-domain watermark embedding.

        For each bit, a block of DCT coefficients in the mid-frequency range
        is perturbed: the mean energy of *dct_coef_count* consecutive
        coefficients is set to either +delta or -delta depending on the bit.
        """
        out = audio.astype(np.float32).copy()
        L = self.frame_length
        cs = self.dct_coef_start
        cc = self.dct_coef_count

        if cs + cc > L:
            raise ValueError(
                "dct_coef_start + dct_coef_count must be <= frame_length."
            )

        for i, bit in enumerate(bit_stream):
            start = i * L
            frame = out[start : start + L]
            coeffs = dct(frame, norm="ortho")

            # Target energy in the chosen coefficient range
            target_region = coeffs[cs : cs + cc]
            current_energy = float(np.mean(np.abs(target_region))) + 1e-9
            delta = self.alpha * current_energy

            # Set coefficients to ±delta (preserving sign)
            sign = np.sign(target_region)
            sign[sign == 0] = 1.0
            polarity = 2.0 * float(bit) - 1.0
            coeffs[cs : cs + cc] = sign * delta * polarity

            out[start : start + L] = idct(coeffs, norm="ortho")

        return np.clip(out, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def required_audio_length(self, message: str) -> int:
        """Return the minimum number of samples needed to embed *message*."""
        bit_stream = self.encoder.encode(message)
        return len(bit_stream) * self.frame_length
