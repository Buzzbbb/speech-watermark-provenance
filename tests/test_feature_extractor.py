"""Tests for FeatureExtractor."""

import numpy as np
import pytest

from src.feature_extractor import FeatureExtractor


SR = 16000


def make_speech(duration: float = 1.0) -> np.ndarray:
    """Synthetic voiced signal (sum of harmonics)."""
    t = np.linspace(0, duration, int(duration * SR), endpoint=False)
    signal = sum(
        (1.0 / k) * np.sin(2 * np.pi * 200 * k * t) for k in range(1, 6)
    )
    signal = signal / np.max(np.abs(signal)) * 0.5
    return signal.astype(np.float32)


class TestFeatureExtractor:
    def setup_method(self):
        self.fe = FeatureExtractor(sample_rate=SR, n_mfcc=13)
        self.audio = make_speech(1.0)

    def test_mfcc_shape(self):
        mfcc = self.fe.mfcc(self.audio, SR)
        assert mfcc.shape[0] == 13
        assert mfcc.shape[1] > 0

    def test_log_mel_shape(self):
        log_mel = self.fe.log_mel_spectrogram(self.audio, SR)
        assert log_mel.shape[0] == self.fe.n_mels
        assert log_mel.shape[1] > 0

    def test_spectral_features_keys(self):
        spec = self.fe.spectral_features(self.audio, SR)
        assert set(spec.keys()) == {"centroid", "bandwidth", "rolloff"}

    def test_zcr_shape(self):
        zcr = self.fe.zero_crossing_rate(self.audio)
        assert zcr.shape[0] == 1

    def test_rms_shape(self):
        rms = self.fe.rms_energy(self.audio)
        assert rms.shape[0] == 1
        assert np.all(rms >= 0)

    def test_f0_shape(self):
        f0 = self.fe.f0(self.audio, SR)
        assert f0.ndim == 1
        assert len(f0) > 0

    def test_extract_all_keys(self):
        features = self.fe.extract_all(self.audio, SR)
        expected = {"mfcc", "log_mel", "centroid", "bandwidth", "rolloff", "zcr", "rms", "f0"}
        assert set(features.keys()) == expected

    def test_speaker_embedding_shape(self):
        emb = self.fe.speaker_embedding(self.audio, SR)
        assert emb.shape == (13,)

    def test_speaker_embedding_same_signal(self):
        """Identical signals must yield similarity = 1.0."""
        emb1 = self.fe.speaker_embedding(self.audio, SR)
        emb2 = self.fe.speaker_embedding(self.audio, SR)
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert abs(sim - 1.0) < 1e-5

    def test_speaker_embedding_different_signals(self):
        """Different signals should yield a similarity < 1.0."""
        t = np.linspace(0, 1.0, SR, endpoint=False)
        noise = self.fe.speaker_embedding(
            np.random.RandomState(0).randn(SR).astype(np.float32), SR
        )
        signal_emb = self.fe.speaker_embedding(self.audio, SR)
        sim = np.dot(signal_emb, noise) / (
            np.linalg.norm(signal_emb) * np.linalg.norm(noise)
        )
        assert sim < 1.0
