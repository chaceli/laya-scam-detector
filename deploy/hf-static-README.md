---
title: Laya 反诈检测 Playground
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: apache-2.0
short_description: 中英诈骗话术风险检测，模型完全在浏览器内运行
---

# Laya 反诈检测 Playground

基于 **微调的 Laya 多语决策模型**的交互式诈骗检测页面。
**推理完全在浏览器内完成** —— 文本不会上传到任何服务器。

## 用法

- **输入文本**：粘贴任意中文/英文消息，或点击预设样本
- **选择输出方式**（Laya 的三种决策原语）：
  - `noul` — 是否诈骗（是/否 + 校准概率）
  - `score` — 风险评分（1-5 期望分 + 各等级分布）
  - `choice` — 诈骗类别（13 类 + 概率分布）
- **高级设置**：可内联编辑问题描述与选项

## 为什么文本不上传

传统做法是把文本发到服务器推理。本页面用
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)（WASM）
在浏览器里直接跑 ONNX 模型，用
[transformers.js](https://huggingface.co/docs/transformers.js) 做分词。
文本从输入到出结果，全程不离开你的设备 —— 对反诈这类敏感场景是实打实的优势。

## 首次加载

首次访问需下载约 **681 MB** 的 fp16 模型（从
[🤗 LiChace/laya-scam-detector-onnx-v3](https://huggingface.co/LiChace/laya-scam-detector-onnx-v3) 拉取）。
浏览器会缓存，之后打开即用。

## 模型能力（600 条中文 holdout）

| 指标 | 数值 |
|---|---|
| is_scam accuracy | 0.967 |
| is_scam precision | 0.989 |
| is_scam recall | 0.957 |
| 13 类 accuracy | 0.810 |

## 技术栈

| 组件 | 选择 |
|---|---|
| 推理 | onnxruntime-web 1.23（WASM）|
| 分词 | @huggingface/transformers 3.7（mmBERT 256k 词表）|
| 模型 | fp16 ONNX，681 MB |
| 托管 | Hugging Face Static Space（无服务端）|

> **为什么不用 int8？** onnxruntime-web 的 WASM int8 算子与原生 ORT 数值不一致
> （同一输入、同一权重下 P(scam) 为 0.194 vs 0.016），而 fp16 与原生差异约 1e-5。
> 详见 [GitHub 仓库](https://github.com/chaceli/laya-scam-detector) 的 `web-static/`。

## 源码

完整项目（微调脚本、数据集构建、评估报告、本地 FastAPI 服务）：
https://github.com/chaceli/laya-scam-detector

## 免责声明

仅供研究与测试。模型判断可能存在误差，概率置信度需按自有数据校准后才可用于生产阈值，
**不可作为唯一决策依据**。
