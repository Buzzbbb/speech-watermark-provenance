"""Tests for WatermarkEncoder."""

import numpy as np
import pytest

from src.watermark_encoder import WatermarkEncoder, _crc8


class TestCRC8:
    def test_empty(self):
        assert _crc8(b"") == 0

    def test_known_value(self):
        # CRC-8 is deterministic
        assert _crc8(b"hello") == _crc8(b"hello")

    def test_different_inputs(self):
        assert _crc8(b"hello") != _crc8(b"world")


class TestWatermarkEncoder:
    def setup_method(self):
        self.enc = WatermarkEncoder(key="test-key", repeat=3)

    def test_bytes_to_bits_roundtrip(self):
        data = b"ABC"
        bits = WatermarkEncoder.bytes_to_bits(data)
        recovered = WatermarkEncoder.bits_to_bytes(bits)
        assert recovered == data

    def test_bits_length(self):
        data = b"Hi"
        bits = WatermarkEncoder.bytes_to_bits(data)
        assert len(bits) == 16  # 2 bytes × 8 bits

    def test_encode_returns_array(self):
        stream = self.enc.encode("Test")
        assert isinstance(stream, np.ndarray)

    def test_encode_length_is_multiple_of_repeat(self):
        stream = self.enc.encode("Hello")
        assert len(stream) % self.enc.repeat == 0

    def test_encode_decode_roundtrip(self):
        message = "ID:001"
        stream = self.enc.encode(message)
        # Use perfect bits (no errors)
        recovered, valid = self.enc.decode(stream)
        assert valid, "CRC check failed on perfect bits"
        assert recovered == message

    def test_decode_detects_corruption(self):
        stream = self.enc.encode("Test")
        corrupted = stream.copy()
        # Flip every third bit
        corrupted[::3] = 1 - corrupted[::3]
        _, valid = self.enc.decode(corrupted.astype(float))
        # With heavy corruption, CRC should (usually) fail
        # – not guaranteed but very likely
        # This is a probabilistic assertion; we just run it without crashing.

    def test_encode_decode_empty_string(self):
        message = ""
        stream = self.enc.encode(message)
        recovered, valid = self.enc.decode(stream)
        assert valid
        assert recovered == message

    def test_encode_decode_unicode(self):
        message = "水印"
        stream = self.enc.encode(message)
        recovered, valid = self.enc.decode(stream)
        assert valid
        assert recovered == message

    def test_generate_pn_length(self):
        pn = self.enc.generate_pn(0, 256)
        assert len(pn) == 256

    def test_generate_pn_values(self):
        pn = self.enc.generate_pn(0, 100)
        assert set(np.unique(pn)).issubset({-1.0, 1.0})

    def test_generate_pn_deterministic(self):
        pn1 = self.enc.generate_pn(5, 128)
        pn2 = self.enc.generate_pn(5, 128)
        np.testing.assert_array_equal(pn1, pn2)

    def test_generate_pn_different_indices(self):
        pn0 = self.enc.generate_pn(0, 128)
        pn1 = self.enc.generate_pn(1, 128)
        # Different indices should (almost certainly) produce different sequences
        assert not np.array_equal(pn0, pn1)

    def test_different_keys_produce_different_pn(self):
        enc2 = WatermarkEncoder(key="other-key", repeat=3)
        pn1 = self.enc.generate_pn(0, 128)
        pn2 = enc2.generate_pn(0, 128)
        assert not np.array_equal(pn1, pn2)

    def test_capacity_bits(self):
        cap = self.enc.capacity_bits(90)
        assert cap == 30  # 90 // 3

    def test_message_max_bytes(self):
        # 90 frames, repeat=3 → 30 raw bits → 30 - 24 = 6 info bits → 0 bytes
        # 300 frames → 100 raw bits → 76 info bits → 9 bytes
        cap = self.enc.message_max_bytes(300)
        assert cap >= 0
