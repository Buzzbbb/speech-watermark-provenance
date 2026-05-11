# 语音模型输出水印与溯源工具

英文名称：`speech-watermark-provenance`

开源地址：`https://github.com/Buzzbbb/speech-watermark-provenance`

项目时间：2025年6月-至今

## 作者信息

- 负责人：林裕斌，专业：网络空间安全，硕士生
- 参与人：曾科，专业：网络空间安全，硕士生
- 参与人：田承金，专业：网络空间安全，硕士生
- 指导教师：吕善翔，网络空间安全学院教师

## 项目内容

本项目针对语音合成与语音转换场景设计输出水印与溯源验证工具，包含音频切片、声学特征提取、水印编码、不可感知嵌入、常见音频处理攻击和水印检测等模块。系统关注水印对语音自然度、说话人相似度和识别准确率的影响，支持输出实验日志与检测报告。项目可用于研究 AI 合成语音标识、音频内容溯源和深度伪造治理中的水印技术，也可扩展到公开语音数据集、自采样本、课堂演示和批量测试。

## 影响力

项目可支撑 AI 合成语音安全、深度伪造治理和音频溯源研究，为课题组开展语音水印实验提供可复用的工程模板。

## 开发语言

Python

## 代码规模

1012行（按当前项目 src/tests/examples 下 Python 代码统计）

## 建议仓库结构

```text
speech-watermark-provenance/
├── README.md
├── LICENSE
├── PROJECT_SUMMARY.md
├── src/
├── examples/
├── tests/
├── docs/
└── screenshots/
```

## 截图材料

- 项目目录截图：`screenshots/directory.png`
- 项目说明截图：`screenshots/readme.png`
- 项目声明截图：`screenshots/license.png`

## 关键词

speech synthesis, audio watermark, provenance, deepfake defense
