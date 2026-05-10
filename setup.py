from setuptools import setup, find_packages

setup(
    name="speech-watermark-provenance",
    version="0.1.0",
    description="语音模型输出水印与溯源工具 – Speech watermarking and provenance verification toolkit",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "soundfile>=0.10.0",
        "librosa>=0.9.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
