"""Tests for AudioSlicer."""

import numpy as np
import pytest
import tempfile
import os

from src.audio_slicer import AudioSlicer


SR = 16000


def make_audio(duration: float = 3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * SR), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestAudioSlicer:
    def setup_method(self):
        self.slicer = AudioSlicer(segment_duration=1.0, sample_rate=SR)

    def test_slice_count(self):
        audio = make_audio(3.0)
        segments = self.slicer.slice(audio, SR)
        # 3-second audio → 3 non-overlapping 1-second segments
        assert len(segments) == 3

    def test_segment_shape(self):
        audio = make_audio(3.0)
        segments = self.slicer.slice(audio, SR)
        for seg, start, end in segments:
            assert len(seg) == SR

    def test_segment_indices(self):
        audio = make_audio(3.0)
        segments = self.slicer.slice(audio, SR)
        assert segments[0][1] == 0
        assert segments[0][2] == SR
        assert segments[1][1] == SR

    def test_partial_segment_included(self):
        # 3.7-second audio – the 0.7-second tail is > 50% of 1.0 s → included
        audio = make_audio(3.7)
        segments = self.slicer.slice(audio, SR)
        assert len(segments) == 4

    def test_partial_segment_excluded(self):
        # 3.3-second audio – the 0.3-second tail is < 50% of 1.0 s → excluded
        audio = make_audio(3.3)
        segments = self.slicer.slice(audio, SR)
        assert len(segments) == 3

    def test_overlapping_slices(self):
        slicer = AudioSlicer(segment_duration=1.0, hop_duration=0.5, sample_rate=SR)
        audio = make_audio(3.0)
        segments = slicer.slice(audio, SR)
        # starts at 0, 0.5, 1.0, 1.5, 2.0 → 5 full segments
        assert len(segments) == 5

    def test_reassemble_length(self):
        audio = make_audio(3.0)
        segments = self.slicer.slice(audio, SR)
        out = self.slicer.reassemble(segments, len(audio))
        assert len(out) == len(audio)

    def test_reassemble_values(self):
        audio = make_audio(3.0)
        segments = self.slicer.slice(audio, SR)
        out = self.slicer.reassemble(segments, len(audio))
        np.testing.assert_allclose(out, audio, atol=1e-5)

    def test_save_load_roundtrip(self):
        audio = make_audio(2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.wav")
            self.slicer.save(path, audio, SR)
            loaded, sr = self.slicer.load(path)
        assert sr == SR
        assert len(loaded) == len(audio)
        np.testing.assert_allclose(loaded, audio, atol=1e-4)

    def test_load_mono_conversion(self):
        """Stereo audio should be converted to mono on load."""
        import soundfile as sf
        stereo = np.column_stack([make_audio(1.0)] * 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stereo.wav")
            sf.write(path, stereo, SR)
            loaded, sr = self.slicer.load(path)
        assert loaded.ndim == 1

    def test_invalid_segment_duration(self):
        slicer = AudioSlicer(segment_duration=0, sample_rate=SR)
        audio = make_audio(1.0)
        with pytest.raises(ValueError):
            slicer.slice(audio, SR)
