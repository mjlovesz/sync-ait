# 大模型推理精度工具（Large Language Model Debug Tool）

## 简介

目前昇腾大模型推理框架主要有 [**加速库(atb)**](../glossary/README.md#at-Ascend-Transformer-Boost) 和 [**torchair**](../glossary/README.md#torchairtorch-图模式)。在推理开发过程中可能会遇到精度问题。

大模型精度调试工具（Large Language Model Debug Tool） 用于帮助开发者快速定位推理开发过程中精度问题，发现根因，提升开发效率。

## 大模型精度调试步骤

大模型精度调试定位，一般思路是从整网到算子，从外到内，从粗到细逐步定位根因，具体定位操作可以视情况调整。一般分为以下 3 个步骤：

