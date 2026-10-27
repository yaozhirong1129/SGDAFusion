# SGDAFusion: Semantic Prior Guided Degradation-Aware Network for Infrared-Visible Image Fusion

> **Information Fusion**  
> Guanghui Yue, Zhirong Yao, Cheng Zhao, Weiqing Yan, Bin Jiang, Tianwei Zhou  
> [[Paper]](https://www.sciencedirect.com/science/article/pii/S1566253526006871)

---

<div align="center">
  <img src="figs/network1.jpg" width="95%" alt="SGDAFusion Overall Architecture">
  <p align="center">
    <em>Fig. 1: Overall architecture of SGDAFusion.</em>
  </p>
</div>

---

## 1. Create Environment

### Create Conda Environment

```bash
conda create -n SGDAFusion python=3.10.13
conda activate SGDAFusion
```

### Install Dependencies

```bash
pip install -r requirements.yml
```

## 2. Prepare Your Dataset

We use the **DA-IVIF** dataset for infrared-visible image fusion. Please download the dataset and organize the files according to the following structure.

<div align="center">
  <img src="figs/dataset.jpg" width="95%" alt="DA-IVIF Dataset">
  <p align="center">
    <em>Fig. 2: DA-IVIF dataset.</em>
  </p>
</div>

The dataset should follow this directory structure:

```text
dataset/
└── your_dataset/
    ├── train/
    │   ├── vis/
    │   └── ir/
    ├── eval/
    │   ├── vis/
    │   └── ir/
    ├── train.csv
    └── test.csv
```

### Dataset Download

- [Google Drive](https://drive.google.com/file/d/1TS0T2wYs06ROuozY9ure2yQrZbUvsdiu/view?usp=sharing)
- [Baidu Yun](https://pan.baidu.com/s/15m9PJEejmfeukk-oYnx42w?pwd=DA26) (Extraction code: `DA26`)

## 3. Download Checkpoints

The pretrained model checkpoints are provided below.

- [Google Drive]()
- [Baidu Yun](https://pan.baidu.com/s/19afZlgi0n4hTfOD2GcFtAw?pwd=SGDA) (Extraction code: `SGDA`)

Download `checkpoints.zip` and extract the files into the project directory. Please ensure that the checkpoint paths match the corresponding paths specified in the configuration files and model-loading code.

## 4. Training Your Model

To train SGDAFusion, run:

```bash
python train.py --gpu 1 --config training.yml
```

### Pretrained Weights

To train the model with DINOv2 and CLIP, download the following pretrained weights:

- **DINOv2:** [`dinov2_vitb14_reg4_pretrain.pth`](https://github.com/facebookresearch/dinov2)
- **CLIP:** [`ViT-B-32.pt`](https://huggingface.co/jinaai/clip-models/tree/main)

Organize the downloaded weights as follows:

```text
SGDAFusion/
├── ViT-B-32.pt
├── dinov2/
│   └── dinov2_vitb14_reg4_pretrain.pth
├── models/
│   └── ...
└── ...
```

## 5. Testing

Run the following command to evaluate the trained model:

```bash
python test.py
```

## 6. Evaluation Metrics

To ensure a fair and unified benchmark, we adopt a PyTorch-based evaluation toolkit to evaluate all comparison methods. We sincerely thank the authors for making their implementation publicly available.

- [PyTorch-based Evaluation Metrics](https://github.com/Topanxal/Image-fusion-evaluation-metrics-code-PyTorch-version-)

## 7. Citation

If you find our work or dataset useful for your research, please cite our paper:

```bibtex
@article{YUE2027104811,
  title     = {Semantic prior guided degradation-aware network for infrared-visible image fusion},
  journal   = {Information Fusion},
  volume    = {139},
  pages     = {104811},
  year      = {2027},
  issn      = {1566-2535},
  doi       = {10.1016/j.inffus.2026.104811},
  url       = {https://www.sciencedirect.com/science/article/pii/S1566253526006871},
  author    = {Guanghui Yue and Zhirong Yao and Cheng Zhao and Weiqing Yan and Bin Jiang and Tianwei Zhou},
  keywords  = {Infrared-visible image fusion, Degradation-aware fusion, Semantic prior, Cross-modal alignment}
}
```
