

# 🖼️ Art Classification with Semi-Supervised and Multimodal Vision Transformers

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-lightgrey)
![Model](https://img.shields.io/badge/model-ViT%20%7C%20ResNet%20%7C%20BLIP-brightgreen)
![Status](https://img.shields.io/badge/status-Completed-green)

## 📌 Overview

This repository explores **art style classification** using deep learning techniques including **semi-supervised learning** with **MixMatch** and **multimodal learning** using **Vision Transformers + BLIP**. The project builds upon the **ArtBench-10** dataset, which contains 60,000 paintings across 10 balanced art styles.

We compare various models—ResNet, EfficientNet, ViT—and demonstrate the strong performance of multimodal fusion models for nuanced artistic understanding.

---

## 🧠 Core Features

- ✅ Supervised Baselines: ResNet-50, EfficientNet-B2, Vision Transformer (ViT)
- 🧪 Semi-Supervised Learning: MixMatch algorithm
- 🎨 Multimodal Fusion: ViT with text embeddings from BLIP
- 📊 Quantitative analysis on classification performance
- 🚀 Transfer learning & fine-tuning from ImageNet weights

---

## 📂 Project Structure

```bash
.
├── vit_resnet_baseline_artwork_classification.ipynb  # Notebook with all experiments
├── Art_Classification_with_Semi_Supervised.pdf       # Full technical report
├── requirements.txt                                  # Python dependencies
├── README.md                                         # This file
````

---

## 📁 Dataset

**[ArtBench-10](https://arxiv.org/abs/2206.11404)**

* 10 artistic styles (e.g., Baroque, Impressionism, Surrealism)
* 60,000 images, balanced (6,000 per class)
* Used for training, validation, and evaluation

---

## 🔧 Training Configuration

* **GPU**: NVIDIA RTX 4090
* **Image Size**: 224 × 224
* **Epochs**: 5
* **Batch Size**: 32
* **Optimizer**: Adam
* **Loss**: CrossEntropy
* **Learning Rate Range**: 1e-2 to 1e-5

---

## 📈 Results Summary

| Model               | Train Acc | Test Acc  |
| ------------------- | --------- | --------- |
| ResNet-50           | 66.8%     | 60.7%     |
| EfficientNet-B2     | 84.9%     | 65.2%     |
| Vision Transformer  | 77.3%     | 65.4%     |
| ViT + MixMatch      | 72.3%     | 54.0%     |
| ViT + BLIP (Fusion) | **94.0%** | **68.4%** |

---

## 🚀 How to Run

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/art-classification-vitm.git
   cd art-classification-vitm
   ```

2. **Install dependencies**

   Included in the notebook

3. **Run the notebook**
   Open `vit_resnet_baseline_artwork_classification.ipynb` in Jupyter or Colab.

---

## 📦 Requirements

```txt
torch>=1.10.0
torchvision
timm
scikit-learn
matplotlib
numpy
pandas
transformers
sentence-transformers
```

---

## 🧪 Tests (Optional)

You can create a `tests/` folder and add minimal unit tests like:

```python
# test_mixmatch.py
def test_mixup_shapes():
    assert mixed_images.shape == original_images.shape
```

## 📘 License

This project is released for educational purposes under the MIT License.

