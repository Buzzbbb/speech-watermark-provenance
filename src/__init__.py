"""
speech-watermark-provenance
===========================
语音模型输出水印与溯源工具

Modules
-------
audio_slicer      : 音频切片 – load, slice and reassemble audio
feature_extractor : 声学特征提取 – MFCC, spectral, pitch features
watermark_encoder : 水印编码 – message ↔ bit-stream with CRC and repetition code
watermark_embedder: 不可感知嵌入 – spread-spectrum and DCT-domain embedding
attack_simulator  : 攻击模拟 – common audio processing attacks
watermark_detector: 水印检测 – correlation-based blind detection
evaluator         : 质量评估 – SNR, LSD, speaker similarity, BER
reporter          : 报告输出 – JSON / CSV experiment logs and detection reports
pipeline          : 端到端流水线 – full watermark embed / detect pipeline
"""

from .audio_slicer import AudioSlicer
from .feature_extractor import FeatureExtractor
from .watermark_encoder import WatermarkEncoder
from .watermark_embedder import WatermarkEmbedder
from .attack_simulator import AttackSimulator
from .watermark_detector import WatermarkDetector
from .evaluator import Evaluator
from .reporter import Reporter
from .pipeline import WatermarkPipeline

__all__ = [
    "AudioSlicer",
    "FeatureExtractor",
    "WatermarkEncoder",
    "WatermarkEmbedder",
    "AttackSimulator",
    "WatermarkDetector",
    "Evaluator",
    "Reporter",
    "WatermarkPipeline",
]
