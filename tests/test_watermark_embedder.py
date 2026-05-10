"""Tests for WatermarkEmbedder."""

import numpy as np
import pytest

from src.watermark_encoder import WatermarkEncoder
from src.watermark_embedder import WatermarkEmbedder


SR = 16000


def make_audio(duration: float = 5.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * SR), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestWatermarkEmbedder:
    def setup_method(self):
        self.encoder = WatermarkEncoder(key="test", repeat=3)
        self.embedder = WatermarkEmbedder(
            encoder=self.encoder, frame_length=256, alpha=0.05, method="ss"
        )
        self.message = "Hi"
        self.audio = make_audio(5.0)

    def test_embed_returns_same_length(self):
        out = self.embedder.embed(self.audio, self.message)
        assert len(out) == len(self.audio)

    def test_embed_output_dtype(self):
        out = self.embedder.embed(self.audio, self.message)
        assert out.dtype == np.float32

    def test_embed_clips_to_unit(self):
        out = self.embedder.embed(self.audio, self.message)
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-6

    def test_embed_modifies_audio(self):
        out = self.embedder.embed(self.audio, self.message)
        assert not np.allclose(out, self.audio)

    def test_embed_snr_reasonable(self):
        """Embedded audio should have SNR > 20 dB with alpha=0.05."""
        out = self.embedder.embed(self.audio, self.message)
        noise = self.audio - out
        signal_power = float(np.mean(self.audio ** 2)) + 1e-12
        noise_power = float(np.mean(noise ** 2)) + 1e-12
        snr = 10 * np.log10(signal_power / noise_power)
        assert snr > 20.0

    def test_embed_dct_method(self):
        emb_dct = WatermarkEmbedder(
            encoder=self.encoder, frame_length=256, alpha=0.05, method="dct"
        )
        out = emb_dct.embed(self.audio, self.message)
        assert len(out) == len(self.audio)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            WatermarkEmbedder(encoder=self.encoder, method="invalid")

    def test_audio_too_short_raises(self):
        short_audio = make_audio(0.01)
        with pytest.raises(ValueError):
            self.embedder.embed(short_audio, "VeryLongMessageThatWillNotFit" * 5)

    def test_required_audio_length(self):
        length = self.embedder.required_audio_length(self.message)
        bit_stream = self.encoder.encode(self.message)
        assert length == len(bit_stream) * 256

    def test_embed_empty_message(self):
        out = self.embedder.embed(self.audio, "")
        assert len(out) == len(self.audio)

    def test_embed_unicode_message(self):
        out = self.embedder.embed(self.audio, "水印")
        assert len(out) == len(self.audio)
