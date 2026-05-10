"""水印编码模块 – WatermarkEncoder

Handles conversion between human-readable messages and the bit-stream that
is physically embedded in audio, including:

- UTF-8 text ↔ binary array
- CRC-8 checksum for integrity verification
- Repetition code for robustness
- Pseudo-random (PN) carrier generation keyed by a secret string
"""

from __future__ import annotations

import hashlib
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CRC-8 (polynomial 0x07, initial value 0x00)
# ---------------------------------------------------------------------------

def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class WatermarkEncoder:
    """Convert messages to watermark bit-streams and back.

    The encoded bit-stream layout (before repetition)::

        [16-bit length] [payload bits] [8-bit CRC]

    where *length* is the number of *payload* bits.

    Parameters
    ----------
    key : str
        Secret key used for PN carrier generation.
    repeat : int
        Repetition factor for each bit (odd number recommended for majority
        voting).  Higher values improve robustness at the cost of capacity.
    """

    def __init__(self, key: str = "swp-default-key", repeat: int = 3) -> None:
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        self.key = key
        self.repeat = repeat

    # ------------------------------------------------------------------
    # Low-level bit conversions
    # ------------------------------------------------------------------

    @staticmethod
    def bytes_to_bits(data: bytes) -> np.ndarray:
        """Convert *data* bytes to an array of 0/1 integers (MSB first)."""
        bits: list[int] = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return np.array(bits, dtype=np.uint8)

    @staticmethod
    def bits_to_bytes(bits: np.ndarray) -> bytes:
        """Convert an array of 0/1 integers back to bytes (MSB first)."""
        bits = bits.astype(np.uint8)
        n_bytes = len(bits) // 8
        out = bytearray(n_bytes)
        for i in range(n_bytes):
            byte = 0
            for b in bits[i * 8 : i * 8 + 8]:
                byte = (byte << 1) | int(b)
            out[i] = byte
        return bytes(out)

    # ------------------------------------------------------------------
    # Public encode / decode
    # ------------------------------------------------------------------

    def encode(self, message: str) -> np.ndarray:
        """Encode *message* into a repeated watermark bit-stream.

        Layout (before repetition):
            16-bit header (payload length in bits) | payload bits | 8-bit CRC

        Parameters
        ----------
        message : str
            UTF-8 text to embed.

        Returns
        -------
        stream : np.ndarray, shape (n_bits,), dtype uint8
            The bit-stream ready for embedding (after repetition).
        """
        payload_bytes = message.encode("utf-8")
        crc = _crc8(payload_bytes)
        frame = payload_bytes + bytes([crc])

        payload_bits = self.bytes_to_bits(frame)
        n_payload = len(payload_bits)

        # 16-bit header stores the number of payload+CRC bits
        header = self.bytes_to_bits(
            bytes([(n_payload >> 8) & 0xFF, n_payload & 0xFF])
        )
        raw_stream = np.concatenate([header, payload_bits])
        return np.repeat(raw_stream, self.repeat).astype(np.uint8)

    def decode(self, stream: np.ndarray) -> Tuple[str, bool]:
        """Decode a watermark bit-stream recovered from audio.

        Applies majority-voting across the repetition code.

        Parameters
        ----------
        stream : np.ndarray
            Soft or hard bit decisions.  Values > 0.5 are treated as '1'.

        Returns
        -------
        message : str
            Decoded text (empty string on failure).
        valid : bool
            True when the CRC check passes.
        """
        stream = np.asarray(stream, dtype=float)
        n_raw = len(stream) // self.repeat
        if n_raw < 16:
            return "", False

        # Majority voting
        usable = stream[: n_raw * self.repeat]
        votes = usable.reshape(n_raw, self.repeat)
        hard_bits = (votes.mean(axis=1) >= 0.5).astype(np.uint8)

        # Parse 16-bit header
        n_payload = int(self.bits_to_bytes(hard_bits[:16]).hex(), 16) if False else 0
        header_bytes = self.bits_to_bytes(hard_bits[:16])
        n_payload = (header_bytes[0] << 8) | header_bytes[1]

        if n_payload <= 0 or 16 + n_payload > n_raw:
            return "", False

        payload_bits = hard_bits[16 : 16 + n_payload]
        if len(payload_bits) < n_payload:
            return "", False

        frame_bytes = self.bits_to_bytes(payload_bits)
        if len(frame_bytes) < 2:
            return "", False

        data_bytes = frame_bytes[:-1]
        expected_crc = frame_bytes[-1]
        actual_crc = _crc8(data_bytes)
        valid = actual_crc == expected_crc

        try:
            message = data_bytes.decode("utf-8")
        except UnicodeDecodeError:
            message = ""
            valid = False

        return message, valid

    # ------------------------------------------------------------------
    # PN carrier generation
    # ------------------------------------------------------------------

    def generate_pn(self, bit_index: int, length: int) -> np.ndarray:
        """Generate a deterministic PN (pseudo-noise) sequence.

        The sequence is derived from the secret *key* and *bit_index* so that
        the same sequence can be reproduced at detection time.

        Parameters
        ----------
        bit_index : int
            Index of the watermark bit (0-based).
        length : int
            Number of samples in the PN sequence.

        Returns
        -------
        pn : np.ndarray, shape (*length*,), values in {-1.0, +1.0}
        """
        seed_str = f"{self.key}:{bit_index}"
        digest = hashlib.sha256(seed_str.encode()).hexdigest()
        seed = int(digest[:8], 16)  # 32-bit seed from first 8 hex chars
        rng = np.random.RandomState(seed)
        return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=length)

    def capacity_bits(self, n_frames: int) -> int:
        """Return the maximum number of *raw* (pre-repetition) bits embeddable.

        Parameters
        ----------
        n_frames : int
            Total number of audio frames available for embedding.
        """
        return n_frames // self.repeat

    def message_max_bytes(self, n_frames: int) -> int:
        """Return the maximum message length in bytes given *n_frames* frames.

        Accounts for the 16-bit header and 8-bit CRC.
        """
        raw_cap = self.capacity_bits(n_frames)
        info_bits = raw_cap - 24  # 16 header + 8 CRC
        return max(0, info_bits // 8)
