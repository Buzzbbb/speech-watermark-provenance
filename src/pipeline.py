"""端到端流水线模块 – WatermarkPipeline

Provides a high-level interface that ties together all sub-modules:
audio loading/slicing → feature extraction → watermark encoding →
imperceptible embedding → attack simulation → watermark detection →
quality evaluation → report generation.

Typical usage::

    pipeline = WatermarkPipeline(key="my-secret", alpha=0.05)

    # Embed
    watermarked, bit_stream = pipeline.embed(audio, sr, message="ID:0001")

    # Detect (blind)
    result = pipeline.detect(watermarked, n_bits=len(bit_stream))

    # Evaluate robustness under all default attacks
    report = pipeline.evaluate_robustness(audio, sr, "ID:0001")

    # Save reports
    pipeline.reporter.save_experiment_log()
    pipeline.reporter.save_detection_report()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio_slicer import AudioSlicer
from .attack_simulator import AttackSimulator
from .evaluator import Evaluator
from .feature_extractor import FeatureExtractor
from .reporter import Reporter
from .watermark_detector import DetectionResult, WatermarkDetector
from .watermark_embedder import WatermarkEmbedder
from .watermark_encoder import WatermarkEncoder


class WatermarkPipeline:
    """End-to-end speech watermarking and provenance verification pipeline.

    Parameters
    ----------
    key : str
        Secret key shared between embedder and detector.
    repeat : int
        Repetition factor for the watermark bit-stream.
    frame_length : int
        Samples per embedding frame.
    alpha : float
        Base watermark strength.
    method : {'ss', 'dct'}
        Embedding algorithm.
    sample_rate : int
        Default sample rate assumed when none is provided.
    output_dir : str
        Directory for saved reports.
    """

    def __init__(
        self,
        key: str = "swp-default-key",
        repeat: int = 3,
        frame_length: int = 256,
        alpha: float = 0.05,
        method: str = "ss",
        sample_rate: int = 16000,
        output_dir: str = "reports",
    ) -> None:
        self.sample_rate = sample_rate

        self.encoder = WatermarkEncoder(key=key, repeat=repeat)
        self.embedder = WatermarkEmbedder(
            encoder=self.encoder,
            frame_length=frame_length,
            alpha=alpha,
            method=method,
        )
        self.detector = WatermarkDetector(
            encoder=self.encoder,
            frame_length=frame_length,
        )
        self.slicer = AudioSlicer(sample_rate=sample_rate)
        self.attacker = AttackSimulator(sample_rate=sample_rate)
        self.evaluator = Evaluator(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        self.reporter = Reporter(output_dir=output_dir)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def embed(
        self,
        audio: np.ndarray,
        sr: int,
        message: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Embed *message* into *audio*.

        Parameters
        ----------
        audio : np.ndarray
        sr : int
        message : str

        Returns
        -------
        watermarked : np.ndarray
        bit_stream : np.ndarray
            The bit-stream that was embedded (used for BER computation).
        """
        watermarked = self.embedder.embed(audio, message)
        bit_stream = self.encoder.encode(message)
        return watermarked, bit_stream

    def detect(
        self,
        audio: np.ndarray,
        n_bits: Optional[int] = None,
    ) -> DetectionResult:
        """Detect the watermark in *audio*.

        Parameters
        ----------
        audio : np.ndarray
        n_bits : int, optional
            Expected bit-stream length.  When None all available frames are
            examined.

        Returns
        -------
        result : DetectionResult
        """
        return self.detector.detect(audio, n_bits=n_bits)

    def embed_from_file(
        self,
        input_path: str,
        output_path: str,
        message: str,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Load audio from file, embed watermark, and save result.

        Parameters
        ----------
        input_path : str
        output_path : str
        message : str

        Returns
        -------
        watermarked : np.ndarray
        sr : int
        bit_stream : np.ndarray
        """
        audio, sr = self.slicer.load(input_path)
        watermarked, bit_stream = self.embed(audio, sr, message)
        self.slicer.save(output_path, watermarked, sr)
        return watermarked, sr, bit_stream

    def detect_from_file(
        self,
        path: str,
        n_bits: Optional[int] = None,
    ) -> DetectionResult:
        """Load audio from file and run detection."""
        audio, _ = self.slicer.load(path)
        return self.detect(audio, n_bits=n_bits)

    # ------------------------------------------------------------------
    # Robustness evaluation
    # ------------------------------------------------------------------

    def evaluate_robustness(
        self,
        audio: np.ndarray,
        sr: int,
        message: str,
        attacks: Optional[List[str]] = None,
        log_results: bool = True,
    ) -> Dict[str, Dict]:
        """Embed a watermark then evaluate it under multiple attacks.

        Parameters
        ----------
        audio : np.ndarray
        sr : int
        message : str
        attacks : list of str, optional
            Attack names to apply.  Defaults to all available attacks.
        log_results : bool
            When True, all results are logged to the internal reporter.

        Returns
        -------
        results : dict mapping attack name → evaluation dict
            Each evaluation dict has keys from :meth:`Evaluator.evaluate`
            plus 'detected', 'decoded_message', and 'confidence'.
        """
        watermarked, bit_stream = self.embed(audio, sr, message)
        n_bits = len(bit_stream)

        # Quality metrics without attack
        no_attack_metrics = self.evaluator.evaluate(
            audio, watermarked, sr=sr
        )
        clean_result = self.detect(watermarked, n_bits=n_bits)
        no_attack_metrics["detected"] = clean_result.detected
        no_attack_metrics["decoded_message"] = clean_result.message
        no_attack_metrics["confidence"] = clean_result.confidence

        if log_results:
            self.reporter.log_experiment(
                no_attack_metrics,
                attack="none",
                message=message,
            )
            self.reporter.log_detection(clean_result, attack="none")

        results: Dict[str, Dict] = {"none": no_attack_metrics}

        attacked_map = self.attacker.apply_all(watermarked, attacks=attacks)
        for attack_name, attacked_audio in attacked_map.items():
            det_result = self.detect(attacked_audio, n_bits=n_bits)
            rec_bits = det_result.bit_stream
            metrics = self.evaluator.evaluate(
                audio,
                attacked_audio,
                sr=sr,
                original_bits=bit_stream,
                recovered_bits=rec_bits,
            )
            metrics["detected"] = det_result.detected
            metrics["decoded_message"] = det_result.message
            metrics["confidence"] = det_result.confidence

            results[attack_name] = metrics

            if log_results:
                self.reporter.log_experiment(
                    metrics, attack=attack_name, message=message
                )
                self.reporter.log_detection(det_result, attack=attack_name)

        return results

    # ------------------------------------------------------------------
    # Feature extraction passthrough
    # ------------------------------------------------------------------

    def extract_features(
        self, audio: np.ndarray, sr: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Extract all acoustic features from *audio*."""
        return self.feature_extractor.extract_all(audio, sr or self.sample_rate)
