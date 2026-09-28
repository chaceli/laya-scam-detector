---
title: Laya 反诈检测 Playground
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: 中文/英文诈骗话术风险检测，基于微调的 Laya 决策模型
---

# Laya 反诈检测 Playground

基于 **微调的 Laya 多语决策模型**（mmBERT-base, 322M，ONNX 本地推理）的交互式诈骗检测页面。

## 用法

- **输入文本**：粘贴任意中文/英文消息，或点击预设样本
- **选择输出方式**（Laya 的三种决策原语）：
  - `noul` — 是否诈骗（是/否 + 概率）
  - `score` — 风险评分（1-5 期望分 + 各等级分布）
  - `choice` — 诈骗类别（13 类 + 概率分布）
- **模型**：本 Space 部署的是微调后的多语模型（中文准确率 0.967）
- **高级设置**：可内联编辑问题描述与选项

## 模型能力（600 条中文 holdout）

| 指标 | 数值 |
|---|---|
| is_scam accuracy | 0.967 |
| is_scam precision | 0.989 |
| is_scam recall | 0.957 |
| 13 类 accuracy | 0.810 |
| p50 延迟 | 125 ms |

## 技术栈

- 推理：ONNX Runtime（无 PyTorch 依赖）
- 模型：`convaiinnovations/laya-multilingual` + LoRA 微调（本仓库微调，Apache-2.0）
- 服务：FastAPI + 原生 HTML/JS
- 模型权重：从 GitHub Release 拉取（fp16，643MB）

## API

- `GET /api/health` — 模型状态
- `GET /api/samples` — 预设样本
- `GET /api/defaults` — 默认问题 schema
- `POST /api/predict` — 推理

## 源码

完整项目（含微调脚本、数据集构建、评估报告）见 GitHub：
https://github.com/chaceli/laya-scam-detector

## 免责声明

本页面仅供研究与测试用途。模型判断可能存在误差，**不可作为唯一决策依据**。
概率置信度需按自有数据校准后才可用于生产阈值。
