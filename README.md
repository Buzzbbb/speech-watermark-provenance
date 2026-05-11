# 语音模型输出水印与溯源工具

`speech-watermark-provenance` 是一个信息隐藏与网络空间安全方向的可运行开源项目，包含核心算法代码、命令行入口、实验配置、示例脚本和 smoke tests。

## Overview

本项目针对语音合成与语音转换场景设计输出水印与溯源验证工具，包含音频切片、声学特征提取、水印编码、不可感知嵌入、常见音频处理攻击和水印检测等模块。系统关注水印对语音自然度、说话人相似度和识别准确率的影响，支持输出实验日志与检测报告。项目可用于研究 AI 合成语音标识、音频内容溯源和深度伪造治理中的水印技术，也可扩展到公开语音数据集、自采样本、课堂演示和批量测试。

## Features

- 统一的数据加载、实验配置和结果保存流程
- 面向信息隐藏/数字水印/隐写分析任务的模块化设计
- 支持实验指标输出、样例结果归档和后续算法扩展
- 适合课程实验、毕业设计、论文复现实验和课题组日常开发

## Quick Start

```bash
python examples/demo.py
python -m unittest discover -s tests
python -m speech_watermark_provenance.cli --message "demo payload" --report docs/cli_report.md
```

## Keywords

speech synthesis · audio watermark · provenance · deepfake defense

## Authors

- 负责人：林裕斌
- 参与人：曾科、田承金
- 指导教师：吕善翔
- 单位：暨南大学网络空间安全学院

## License

本项目建议采用 MIT License；实际开源时请根据课题组要求确认许可证。
