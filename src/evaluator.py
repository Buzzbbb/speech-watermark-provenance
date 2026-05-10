"""质量评估模块 – Evaluator

Measures the impact of watermark embedding on perceptual audio quality,
speaker identity preservation, and watermark robustness.

Metrics
-------
SNR             : Signal-to-Noise Ratio (dB) – imperceptibility proxy.
LSD             : Log-Spectral Distortion (dB) – spectral fidelity proxy.
speaker_sim     : Cosine similarity of mean MFCC vectors – speaker-identity proxy.
pesq_approx     : Simplified PESQ approximation based on SNR and LSD.
ber             : Bit Error Rate – watermark robustness after attack.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.fft import fft

from .feature_extractor import FeatureExtractor


class Evaluator:
    """Compute quality and robustness metrics for watermarked audio.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signals.
    n_fft : int
        FFT size for spectral metrics.
    hop_length : int
        Hop length for MFCC computation.
    n_mfcc : int
        Number of MFCC coefficients used for speaker similarity.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        n_mfcc: int = 13,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._extractor = FeatureExtractor(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    # ------------------------------------------------------------------
    # Imperceptibility metrics
    # ------------------------------------------------------------------

    def snr(self, original: np.ndarray, watermarked: np.ndarray) -> float:
        """Signal-to-Noise Ratio in dB.

        Higher values indicate better imperceptibility.

        Parameters
        ----------
        original : np.ndarray
            Clean (un-watermarked) audio.
        watermarked : np.ndarray
            Watermarked audio (same length as *original*).

        Returns
        -------
        snr_db : float
        """
        n = min(len(original), len(watermarked))
        signal_power = float(np.mean(original[:n] ** 2)) + 1e-12
        noise = original[:n] - watermarked[:n]
        noise_power = float(np.mean(noise ** 2)) + 1e-12
        return 10.0 * np.log10(signal_power / noise_power)

    def lsd(self, original: np.ndarray, watermarked: np.ndarray) -> float:
        """Log-Spectral Distortion in dB.

        Lower values indicate better spectral fidelity.

        Parameters
        ----------
        original : np.ndarray
        watermarked : np.ndarray

        Returns
        -------
        lsd_db : float
        """
        n = min(len(original), len(watermarked))
        orig = original[:n]
        wmk = watermarked[:n]
        n_frames = n // self.n_fft
        if n_frames == 0:
            return float("inf")

        lsd_total = 0.0
        for k in range(n_frames):
            o_frame = orig[k * self.n_fft : (k + 1) * self.n_fft]
            w_frame = wmk[k * self.n_fft : (k + 1) * self.n_fft]
            O = np.abs(fft(o_frame)) ** 2 + 1e-12
            W = np.abs(fft(w_frame)) ** 2 + 1e-12
            lsd_frame = float(np.sqrt(np.mean((10.0 * np.log10(O / W)) ** 2)))
            lsd_total += lsd_frame

        return lsd_total / n_frames

    def pesq_approx(self, original: np.ndarray, watermarked: np.ndarray) -> float:
        """Simplified PESQ approximation (MOS-scale 1–4.5).

        Uses a heuristic mapping of SNR and LSD to an approximate MOS value.
        This is **not** a standards-compliant PESQ implementation; it serves
        as a lightweight proxy for research and logging purposes.

        Returns
        -------
        mos : float in [1.0, 4.5]
        """
        snr_val = self.snr(original, watermarked)
        lsd_val = self.lsd(original, watermarked)
        # Clamp inputs to reasonable ranges
        snr_norm = min(max(snr_val / 40.0, 0.0), 1.0)   # 40 dB → 1.0
        lsd_norm = min(max(lsd_val / 10.0, 0.0), 1.0)   # 10 dB → bad
        mos = 1.0 + 3.5 * snr_norm * (1.0 - 0.8 * lsd_norm)
        return float(np.clip(mos, 1.0, 4.5))

    # ------------------------------------------------------------------
    # Speaker similarity
    # ------------------------------------------------------------------

    def speaker_similarity(
        self,
        original: np.ndarray,
        watermarked: np.ndarray,
        sr: Optional[int] = None,
    ) -> float:
        """Cosine similarity of mean MFCC vectors.

        A value close to 1.0 indicates high speaker identity preservation.

        Parameters
        ----------
        original : np.ndarray
        watermarked : np.ndarray
        sr : int, optional

        Returns
        -------
        similarity : float in [-1, 1]
        """
        sr = sr or self.sample_rate
        emb_orig = self._extractor.speaker_embedding(original, sr)
        emb_wmk = self._extractor.speaker_embedding(watermarked, sr)
        norm = np.linalg.norm(emb_orig) * np.linalg.norm(emb_wmk) + 1e-12
        return float(np.dot(emb_orig, emb_wmk) / norm)

    # ------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------

    def ber(
        self, original_bits: np.ndarray, recovered_bits: np.ndarray
    ) -> float:
        """Bit Error Rate between *original_bits* and *recovered_bits*.

        Returns
        -------
        ber : float in [0, 1]  – lower is better
        """
        n = min(len(original_bits), len(recovered_bits))
        if n == 0:
            return 1.0
        return float(np.sum(original_bits[:n] != recovered_bits[:n])) / n

    # ------------------------------------------------------------------
    # Aggregate evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        original: np.ndarray,
        watermarked: np.ndarray,
        sr: Optional[int] = None,
        original_bits: Optional[np.ndarray] = None,
        recovered_bits: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all quality metrics and return a summary dictionary.

        Parameters
        ----------
        original : np.ndarray
        watermarked : np.ndarray
        sr : int, optional
        original_bits : np.ndarray, optional
            If provided along with *recovered_bits*, BER is computed.
        recovered_bits : np.ndarray, optional

        Returns
        -------
        metrics : dict with keys 'snr_db', 'lsd_db', 'pesq_approx',
                  'speaker_similarity', and optionally 'ber'.
        """
        metrics: Dict[str, float] = {
            "snr_db": self.snr(original, watermarked),
            "lsd_db": self.lsd(original, watermarked),
            "pesq_approx": self.pesq_approx(original, watermarked),
            "speaker_similarity": self.speaker_similarity(original, watermarked, sr),
        }
        if original_bits is not None and recovered_bits is not None:
            metrics["ber"] = self.ber(original_bits, recovered_bits)
        return metrics
