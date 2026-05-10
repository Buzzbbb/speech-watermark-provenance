"""报告输出模块 – Reporter

Serialises experiment results and detection outcomes to structured formats
(JSON and CSV) for logging, analysis, and audit trails.

Usage example::

    reporter = Reporter(output_dir="reports")
    reporter.log_experiment(metrics, attack="noise", snr_db=20.0)
    reporter.write_detection_report(detection_result, audio_path="sample.wav")
    reporter.save_experiment_log()
    reporter.save_detection_report()
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .watermark_detector import DetectionResult


class Reporter:
    """Collect and persist experiment logs and detection reports.

    Parameters
    ----------
    output_dir : str
        Directory where log files will be written.
    run_id : str, optional
        Human-readable run identifier inserted into file names and records.
        Defaults to an ISO-8601 timestamp.
    """

    def __init__(
        self,
        output_dir: str = "reports",
        run_id: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        self._experiment_log: List[Dict[str, Any]] = []
        self._detection_report: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Experiment logging
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        metrics: Dict[str, float],
        *,
        attack: str = "none",
        message: str = "",
        audio_path: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an experiment record to the in-memory log.

        Parameters
        ----------
        metrics : dict
            Quality metrics dict as returned by :meth:`Evaluator.evaluate`.
        attack : str
            Name of the attack applied (or 'none').
        message : str
            Watermark message that was embedded.
        audio_path : str
            Path to the source audio file.
        extra : dict, optional
            Additional key-value pairs to include in the record.
        """
        record: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "audio_path": audio_path,
            "message": message,
            "attack": attack,
            **metrics,
        }
        if extra:
            record.update(extra)
        self._experiment_log.append(record)

    def save_experiment_log(
        self,
        filename: Optional[str] = None,
        fmt: str = "json",
    ) -> str:
        """Write the accumulated experiment log to disk.

        Parameters
        ----------
        filename : str, optional
            Output file name (without directory prefix).  Defaults to
            ``experiment_log_{run_id}.{fmt}``.
        fmt : {'json', 'csv'}
            Output format.

        Returns
        -------
        path : str
            Absolute path of the written file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename or f"experiment_log_{self.run_id}.{fmt}"
        path = os.path.join(self.output_dir, filename)

        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._experiment_log, f, ensure_ascii=False, indent=2)
        elif fmt == "csv":
            if self._experiment_log:
                keys = list(self._experiment_log[0].keys())
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(self._experiment_log)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'json' or 'csv'.")

        return os.path.abspath(path)

    # ------------------------------------------------------------------
    # Detection reporting
    # ------------------------------------------------------------------

    def log_detection(
        self,
        result: DetectionResult,
        *,
        audio_path: str = "",
        attack: str = "none",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a detection result to the in-memory detection report.

        Parameters
        ----------
        result : DetectionResult
        audio_path : str
        attack : str
        extra : dict, optional
        """
        record: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "audio_path": audio_path,
            "attack": attack,
            "detected": result.detected,
            "message": result.message,
            "confidence": round(result.confidence, 6),
            "n_frames_used": result.n_frames_used,
        }
        if extra:
            record.update(extra)
        self._detection_report.append(record)

    def save_detection_report(
        self,
        filename: Optional[str] = None,
        fmt: str = "json",
    ) -> str:
        """Write the accumulated detection report to disk.

        Parameters
        ----------
        filename : str, optional
        fmt : {'json', 'csv'}

        Returns
        -------
        path : str
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename or f"detection_report_{self.run_id}.{fmt}"
        path = os.path.join(self.output_dir, filename)

        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._detection_report, f, ensure_ascii=False, indent=2)
        elif fmt == "csv":
            if self._detection_report:
                keys = list(self._detection_report[0].keys())
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(self._detection_report)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'json' or 'csv'.")

        return os.path.abspath(path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset in-memory log and report."""
        self._experiment_log.clear()
        self._detection_report.clear()

    def summary(self) -> Dict[str, Any]:
        """Return a high-level summary of the current in-memory data."""
        return {
            "run_id": self.run_id,
            "n_experiments": len(self._experiment_log),
            "n_detections": len(self._detection_report),
            "detection_rate": (
                sum(r["detected"] for r in self._detection_report)
                / len(self._detection_report)
                if self._detection_report
                else None
            ),
        }
