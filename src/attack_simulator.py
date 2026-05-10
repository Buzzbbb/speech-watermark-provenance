"""攻击模拟模块 – AttackSimulator

Simulates common audio processing operations that an adversary (or lossy
transmission channel) might apply to a watermarked audio signal.

Supported attacks
-----------------
noise          : Additive white Gaussian noise at a specified SNR level.
lowpass        : Low-pass FIR filter.
highpass       : High-pass FIR filter.
resample       : Downsample then upsample (mimics codec resampling).
amplitude      : Random amplitude scaling.
crop           : Trim a fraction of samples from both ends and re-pad.
quantize       : Reduce bit-depth (simulates lossy compression).
time_stretch   : Time-scale modification via nearest-sample interpolation.
echo           : Add a delayed, attenuated copy of the signal.
bandpass       : Band-pass FIR filter.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly
from math import gcd


class AttackSimulator:
    """Apply audio processing attacks to watermarked signals.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio signals that will be processed.
    seed : int, optional
        Random seed for reproducible stochastic attacks.
    """

    def __init__(self, sample_rate: int = 16000, seed: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Individual attacks
    # ------------------------------------------------------------------

    def noise(self, audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Add white Gaussian noise at *snr_db* dB above the signal power.

        Parameters
        ----------
        audio : np.ndarray
        snr_db : float
            Signal-to-noise ratio in dB.  Lower values → stronger noise.

        Returns
        -------
        noisy : np.ndarray
        """
        signal_power = float(np.mean(audio ** 2)) + 1e-12
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = self._rng.normal(0, np.sqrt(noise_power), size=audio.shape)
        return np.clip(audio + noise.astype(np.float32), -1.0, 1.0)

    def lowpass(
        self, audio: np.ndarray, cutoff_hz: float = 4000.0, num_taps: int = 101
    ) -> np.ndarray:
        """Apply a low-pass FIR filter.

        Parameters
        ----------
        cutoff_hz : float
            Filter cut-off frequency in Hz.
        num_taps : int
            Number of FIR taps (must be odd).
        """
        nyq = self.sample_rate / 2.0
        cutoff = min(cutoff_hz / nyq, 0.99)
        b = firwin(num_taps, cutoff)
        return np.clip(lfilter(b, 1.0, audio).astype(np.float32), -1.0, 1.0)

    def highpass(
        self, audio: np.ndarray, cutoff_hz: float = 300.0, num_taps: int = 101
    ) -> np.ndarray:
        """Apply a high-pass FIR filter.

        Parameters
        ----------
        cutoff_hz : float
            Filter cut-off frequency in Hz.
        """
        nyq = self.sample_rate / 2.0
        cutoff = min(cutoff_hz / nyq, 0.99)
        b = firwin(num_taps, cutoff, pass_zero=False)
        return np.clip(lfilter(b, 1.0, audio).astype(np.float32), -1.0, 1.0)

    def bandpass(
        self,
        audio: np.ndarray,
        low_hz: float = 300.0,
        high_hz: float = 3400.0,
        num_taps: int = 101,
    ) -> np.ndarray:
        """Apply a band-pass FIR filter (telephone-band approximation)."""
        nyq = self.sample_rate / 2.0
        low = min(low_hz / nyq, 0.99)
        high = min(high_hz / nyq, 0.99)
        b = firwin(num_taps, [low, high], pass_zero=False)
        return np.clip(lfilter(b, 1.0, audio).astype(np.float32), -1.0, 1.0)

    def resample(
        self, audio: np.ndarray, target_sr: int = 8000
    ) -> np.ndarray:
        """Downsample to *target_sr* then upsample back to *sample_rate*.

        Simulates codec-induced resampling artefacts.

        Parameters
        ----------
        target_sr : int
            Intermediate sample rate (must be < sample_rate).
        """
        if target_sr >= self.sample_rate:
            return audio.copy()
        g = gcd(target_sr, self.sample_rate)
        down = self.sample_rate // g
        up = target_sr // g
        # downsample
        downsampled = resample_poly(audio, up, down)
        # upsample back
        up2 = down
        down2 = up
        resampled = resample_poly(downsampled, up2, down2)
        # Align length
        n = len(audio)
        if len(resampled) >= n:
            return resampled[:n].astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        out[: len(resampled)] = resampled
        return out

    def amplitude(
        self, audio: np.ndarray, scale_range: Tuple[float, float] = (0.7, 1.3)
    ) -> np.ndarray:
        """Scale amplitude by a random factor in *scale_range*."""
        scale = self._rng.uniform(*scale_range)
        return np.clip((audio * scale).astype(np.float32), -1.0, 1.0)

    def crop(self, audio: np.ndarray, crop_fraction: float = 0.05) -> np.ndarray:
        """Remove *crop_fraction* of samples from each end and zero-pad back.

        Parameters
        ----------
        crop_fraction : float
            Fraction of total length to remove from each side (0–0.4).
        """
        crop_fraction = min(max(crop_fraction, 0.0), 0.4)
        n = len(audio)
        trim = int(n * crop_fraction)
        if trim == 0:
            return audio.copy()
        cropped = audio[trim : n - trim]
        out = np.zeros(n, dtype=np.float32)
        out[trim : trim + len(cropped)] = cropped
        return out

    def quantize(self, audio: np.ndarray, bits: int = 8) -> np.ndarray:
        """Reduce bit-depth to *bits* bits (uniform quantisation).

        Parameters
        ----------
        bits : int
            Target bit depth (1–16).
        """
        bits = max(1, min(bits, 16))
        levels = 2 ** bits
        quantized = np.round(audio * (levels / 2)) / (levels / 2)
        return np.clip(quantized.astype(np.float32), -1.0, 1.0)

    def time_stretch(
        self, audio: np.ndarray, rate: float = 1.1
    ) -> np.ndarray:
        """Stretch or compress time by *rate* using nearest-sample interpolation.

        Parameters
        ----------
        rate : float
            > 1.0 → faster (shorter output); < 1.0 → slower (longer output).
            Output is trimmed or zero-padded to match the input length.
        """
        if rate <= 0:
            raise ValueError("rate must be > 0")
        n = len(audio)
        stretched_len = max(1, int(n / rate))
        indices = np.linspace(0, n - 1, stretched_len)
        stretched = np.interp(indices, np.arange(n), audio).astype(np.float32)
        if len(stretched) >= n:
            return stretched[:n]
        out = np.zeros(n, dtype=np.float32)
        out[: len(stretched)] = stretched
        return out

    def echo(
        self,
        audio: np.ndarray,
        delay_ms: float = 50.0,
        decay: float = 0.3,
    ) -> np.ndarray:
        """Add a single delayed echo.

        Parameters
        ----------
        delay_ms : float
            Echo delay in milliseconds.
        decay : float
            Amplitude of the echo relative to the original (0–1).
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000.0)
        if delay_samples == 0:
            return audio.copy()
        out = audio.astype(np.float32).copy()
        if delay_samples < len(audio):
            out[delay_samples:] += decay * audio[: len(audio) - delay_samples]
        return np.clip(out, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Batch / combined
    # ------------------------------------------------------------------

    def apply(self, audio: np.ndarray, attack: str, **kwargs) -> np.ndarray:
        """Apply a named attack.

        Parameters
        ----------
        audio : np.ndarray
        attack : str
            One of the attack names listed in the module docstring.
        **kwargs
            Forwarded to the corresponding attack method.

        Returns
        -------
        attacked : np.ndarray
        """
        methods: Dict[str, object] = {
            "noise": self.noise,
            "lowpass": self.lowpass,
            "highpass": self.highpass,
            "bandpass": self.bandpass,
            "resample": self.resample,
            "amplitude": self.amplitude,
            "crop": self.crop,
            "quantize": self.quantize,
            "time_stretch": self.time_stretch,
            "echo": self.echo,
        }
        if attack not in methods:
            raise ValueError(
                f"Unknown attack '{attack}'. Available: {sorted(methods)}"
            )
        return methods[attack](audio, **kwargs)  # type: ignore[operator]

    def apply_all(
        self,
        audio: np.ndarray,
        attacks: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Apply multiple attacks and return a dict mapping name → attacked audio.

        Parameters
        ----------
        audio : np.ndarray
        attacks : list of str, optional
            Attack names to apply.  Defaults to all available attacks with
            default parameters.
        """
        defaults: Dict[str, Dict] = {
            "noise": {"snr_db": 20.0},
            "lowpass": {"cutoff_hz": 4000.0},
            "highpass": {"cutoff_hz": 300.0},
            "bandpass": {"low_hz": 300.0, "high_hz": 3400.0},
            "resample": {"target_sr": 8000},
            "amplitude": {"scale_range": (0.7, 1.3)},
            "crop": {"crop_fraction": 0.05},
            "quantize": {"bits": 8},
            "time_stretch": {"rate": 1.1},
            "echo": {"delay_ms": 50.0, "decay": 0.3},
        }
        selected = attacks if attacks is not None else list(defaults)
        return {name: self.apply(audio, name, **defaults[name]) for name in selected}
