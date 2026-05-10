"""音频切片模块 – AudioSlicer

Provides utilities to load, slice, and save audio files for watermark
processing.  All audio is handled as 32-bit floating-point mono arrays
normalised to the range [-1, 1].
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf


class AudioSlicer:
    """Load, slice, and save audio for watermark processing.

    Parameters
    ----------
    segment_duration : float
        Length of each slice in seconds.
    hop_duration : float, optional
        Hop between slice starts in seconds.  Defaults to *segment_duration*
        (non-overlapping slices).
    sample_rate : int
        Target sample rate used when synthesising silence padding.
    """

    def __init__(
        self,
        segment_duration: float = 1.0,
        hop_duration: Optional[float] = None,
        sample_rate: int = 16000,
    ) -> None:
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration if hop_duration is not None else segment_duration
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self, path: str) -> Tuple[np.ndarray, int]:
        """Load an audio file and return a mono float32 array with its sample rate.

        Parameters
        ----------
        path : str
            Path to the audio file (WAV, FLAC, OGG, …).

        Returns
        -------
        audio : np.ndarray, shape (N,), dtype float32
        sample_rate : int
        """
        audio, sr = sf.read(path, always_2d=False, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr

    def save(self, path: str, audio: np.ndarray, sr: int) -> None:
        """Write *audio* to *path*.

        Parameters
        ----------
        path : str
            Destination path (extension determines format).
        audio : np.ndarray
            Mono float32 audio array.
        sr : int
            Sample rate.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sf.write(path, audio.astype(np.float32), sr)

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------

    def slice(
        self, audio: np.ndarray, sr: int
    ) -> List[Tuple[np.ndarray, int, int]]:
        """Divide *audio* into fixed-length segments.

        The last incomplete segment is zero-padded and included only when it
        contains at least 50 % of the nominal segment length.

        Parameters
        ----------
        audio : np.ndarray
            Mono audio array.
        sr : int
            Sample rate of *audio*.

        Returns
        -------
        segments : list of (segment, start_sample, end_sample)
            Each element is a tuple containing the audio segment array, the
            inclusive start sample index, and the exclusive end sample index
            within *audio*.
        """
        seg_len = int(self.segment_duration * sr)
        hop_len = int(self.hop_duration * sr)
        if seg_len <= 0:
            raise ValueError("segment_duration must be > 0.")
        if hop_len <= 0:
            raise ValueError("hop_duration must be > 0.")

        segments: List[Tuple[np.ndarray, int, int]] = []
        start = 0
        while start + seg_len <= len(audio):
            segments.append((audio[start : start + seg_len].copy(), start, start + seg_len))
            start += hop_len

        # Include trailing partial segment (if > 50 % full)
        remainder = len(audio) - start
        if 0 < remainder > seg_len * 0.5:
            padded = np.zeros(seg_len, dtype=audio.dtype)
            padded[:remainder] = audio[start : start + remainder]
            segments.append((padded, start, len(audio)))

        return segments

    def reassemble(
        self,
        segments: List[Tuple[np.ndarray, int, int]],
        total_length: int,
    ) -> np.ndarray:
        """Merge processed segments back into a single array.

        Overlapping regions are averaged.

        Parameters
        ----------
        segments : list of (segment, start_sample, end_sample)
            Segments as returned by :meth:`slice` (possibly modified).
        total_length : int
            Length of the output array in samples.

        Returns
        -------
        audio : np.ndarray, shape (*total_length*,), dtype float32
        """
        output = np.zeros(total_length, dtype=np.float32)
        weight = np.zeros(total_length, dtype=np.float32)

        for seg, start, end in segments:
            seg_len = min(len(seg), total_length - start)
            output[start : start + seg_len] += seg[:seg_len]
            weight[start : start + seg_len] += 1.0

        non_zero = weight > 0
        output[non_zero] /= weight[non_zero]
        return output
