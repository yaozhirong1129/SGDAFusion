# SGDAFusion: Semantic Prior Guided Degradation-Aware Network for Infrared-Visible Image Fusion

> **Information Fusion 2026**  
> Guanghui Yue, Zhirong Yao, Cheng Zhao, Weiqing Yan, Bin Jiang, Tianwei Zhou*  
> [(Paper)]

---

<div align="center">
  <img src="figs/network1.jpg" width="95%" alt="SGDAFusion Overall Architecture">
  <p align="center">
    <em>Fig 1: SGDAFusion network。</em>
  </p>
</div>

---

## 1. Create Environment
- Create Conda Environment

```bash
conda create -n SGDAFusion python=3.10.13
conda activate SGDAFusion
```

- Install Dependencies

```bash
pip install -r requirements.yml
```

## 2. Prepare Your Dataset
---

<div align="center">
  <img src="figs/dataset.jpg" width="95%" alt="dataset">
  <p align="center">
    <em>Fig 1: DA-IVIF</em>
  </p>
</div>

---

```bash
git clone [https://github.com/YourUsername/SGDAFusion.git](https://github.com/YourUsername/SGDAFusion.git)
cd SGDAFusion
pip install -r requirements.txt
