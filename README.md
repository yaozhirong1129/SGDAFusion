# SGDAFusion: Semantic Prior Guided Degradation-Aware Network for Infrared-Visible Image Fusion

> **Information Fusion 2026**  
> 作者姓名, 作者姓名, 通讯作者*  
> [论文链接 (Paper)] | [补充材料 (Supplementary)] | [预训练权重 (Weights)]

---

<div align="center">
  <img src="assets/framework.png" width="95%" alt="SGDAFusion Overall Architecture">
  <p align="center">
    <em>图 1: SGDAFusion 整体网络框架图。</em>
  </p>
</div>

---

## 动态更新 (News)

- **[2026.xx]** 论文被 **Information Fusion** 正式录用。
- **[2026.xx]** 开源代码与预训练模型已发布。

## 简介 (Overview)

红外与可见光图像融合（IVIF）在复杂恶劣场景下面临退化（如噪声、过曝光、欠曝光、模糊）以及语义缺失等挑战。本文提出了 **SGDAFusion**，通过结合语义先验指导与退化感知机制，有效抑制图像退化干扰并增强关键目标与场景的语义表达。

## 核心特性 (Key Features)

- **退化感知机制 (Degradation-Aware Modeling)**：自适应识别并补偿源图像中的各种退化因素。
- **语义先验引导 (Semantic Prior Guidance)**：引入高级语义特征，提升下游任务（如目标检测、语义分割）的融合质量。
- **优越的融合性能 (State-of-the-Art Performance)**：在多个公开基准数据集上均取得了更优的视觉质量与定量指标。

---

## 框架图插入说明 (Figure Placement)

在 GitHub 仓库中展示框架图的操作步骤：
1. 在仓库根目录下创建 `assets/` 文件夹。
2. 将论文框架图重命名为 `framework.png` 并放置在 `assets/` 目录下。
3. 提交并推送到 GitHub 仓库即可自动显示。

---

## 环境配置 (Environment Setup)

### 1. 基础依赖

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.8

### 2. 安装依赖包

```bash
git clone [https://github.com/YourUsername/SGDAFusion.git](https://github.com/YourUsername/SGDAFusion.git)
cd SGDAFusion
pip install -r requirements.txt
