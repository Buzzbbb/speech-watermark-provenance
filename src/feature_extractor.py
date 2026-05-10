"""声学特征提取模块 – FeatureExtractor

Extracts acoustic features used for quality evaluation (speaker similarity,
spectral distortion) and general analysis.

Features
--------
- MFCC (Mel-frequency cepstral coefficients)
- Log-Mel spectrogram
- Spectral centroid, bandwidth, and roll-off
- Zero-crossing rate
- RMS energy
- Fundamental frequency (F0) via autocorrelation
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import librosa


class FeatureExtractor:
    """Extract acoustic features from a mono audio array.

    Parameters
    ----------
    sample_rate : int
        Expected sample rate of input audio.
    n_mfcc : int
        Number of MFCC coefficients to compute.
    n_fft : int
        FFT window length in samples.
    hop_length : int
        FFT hop length in samples.
    n_mels : int
        Number of Mel filter-bank channels.
    f0_min : float
        Minimum fundamental frequency for F0 estimation (Hz).
    f0_max : float
        Maximum fundamental frequency for F0 estimation (Hz).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 128,
        n_mels: int = 40,
        f0_min: float = 60.0,
        f0_max: float = 400.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f0_min = f0_min
        self.f0_max = f0_max

    # ------------------------------------------------------------------
    # Individual feature methods
    # ------------------------------------------------------------------

    def mfcc(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Compute MFCC matrix.

        Returns
        -------
        mfcc : np.ndarray, shape (n_mfcc, T)
        """
        sr = sr or self.sample_rate
        return librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

    def log_mel_spectrogram(
        self, audio: np.ndarray, sr: Optional[int] = None
    ) -> np.ndarray:
        """Compute log-Mel spectrogram.

        Returns
        -------
        log_mel : np.ndarray, shape (n_mels, T)
        """
        sr = sr or self.sample_rate
        mel = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        return librosa.power_to_db(mel, ref=np.max)

    def spectral_features(
        self, audio: np.ndarray, sr: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Compute spectral centroid, bandwidth, and roll-off.

        Returns
        -------
        dict with keys 'centroid', 'bandwidth', 'rolloff' (each shape (1, T))
        """
        sr = sr or self.sample_rate
        y = audio.astype(np.float32)
        return {
            "centroid": librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            ),
            "bandwidth": librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            ),
            "rolloff": librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            ),
        }

    def zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Compute per-frame zero-crossing rate.

        Returns
        -------
        zcr : np.ndarray, shape (1, T)
        """
        return librosa.feature.zero_crossing_rate(
            audio.astype(np.float32), hop_length=self.hop_length
        )

    def rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """Compute per-frame RMS energy.

        Returns
        -------
        rms : np.ndarray, shape (1, T)
        """
        return librosa.feature.rms(
            y=audio.astype(np.float32), hop_length=self.hop_length
        )

    def f0(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Estimate fundamental frequency (F0) via autocorrelation.

        Returns
        -------
        f0_contour : np.ndarray, shape (T,)  – Hz, 0.0 for unvoiced frames
        """
        sr = sr or self.sample_rate
        f0_vals, _, _ = librosa.pyin(
            audio.astype(np.float32),
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=sr,
            hop_length=self.hop_length,
        )
        f0_vals = np.where(np.isnan(f0_vals), 0.0, f0_vals)
        return f0_vals

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def extract_all(
        self, audio: np.ndarray, sr: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Extract all features and return a unified dictionary.

        Returns
        -------
        features : dict
            Keys: 'mfcc', 'log_mel', 'centroid', 'bandwidth', 'rolloff',
                  'zcr', 'rms', 'f0'
        """
        sr = sr or self.sample_rate
        spec = self.spectral_features(audio, sr)
        return {
            "mfcc": self.mfcc(audio, sr),
            "log_mel": self.log_mel_spectrogram(audio, sr),
            "centroid": spec["centroid"],
            "bandwidth": spec["bandwidth"],
            "rolloff": spec["rolloff"],
            "zcr": self.zero_crossing_rate(audio),
            "rms": self.rms_energy(audio),
            "f0": self.f0(audio, sr),
        }

    def speaker_embedding(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Return a simple speaker embedding: mean MFCC vector (n_mfcc,).

        This is a lightweight approximation suitable for cosine-similarity
        comparisons without requiring a neural speaker model.
        """
        return self.mfcc(audio, sr).mean(axis=1)
