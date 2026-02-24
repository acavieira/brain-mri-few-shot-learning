# Brain MRI Few-Shot Learning

This repository implements an episodic Few-Shot Learning (FSL) framework for brain tumor classification using Prototypical Networks in PyTorch.

The project explores metric-based learning under low-data regimes (4-way K-shot setup) applied to medical MRI classification.

---

## Research Context

Brain tumor classification in MRI is challenging when labeled data is limited. Few-Shot Learning (FSL) allows models to generalize from a small number of examples per class.

This implementation includes:

- 4-way K-shot episodic training
- Prototypical Networks
- Euclidean distance-based classification
- Timestamped model checkpoint saving
- CPU/GPU compatibility

---

## Dataset

This repository includes the **Brain Tumor MRI Dataset** originally published on Kaggle:

Source:  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Original author: **Masoud Nickparvar**

The dataset is licensed under terms that allow:

- Sharing and redistribution
- Adaptation and modification
- Commercial and research use

All credit for the dataset belongs to the original author.

---

## Project Structure

- `src/`
  - `fsl_protonet/`
    - `__init__.py`
    - `config.py`
    - `data.py`
    - `models.py`
    - `protonet.py`
    - `utils.py`
    - `engine.py`

- `scripts/`
  - `train.py`
  - `infer_episode.py`

- `data/`
  - `train/`
    - `glioma/`
    - `meningioma/`
    - `pituitary/`
    - `no_tumor/`
  - `test/`
    - `glioma/`
    - `meningioma/`
    - `pituitary/`
    - `no_tumor/`

- `experiments/`
- `requirements.txt`
- `README.md`
- `LICENSE`

---

## Installation

Make sure you are in the root directory of the project.

### 1. Create a virtual environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

Make sure you are in the root directory:

```
FSL_Protonet/
```

---

### Train the model

macOS / Linux:

```bash
PYTHONPATH=src python scripts/train.py
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts/train.py
```

During training, the following experiments will be saved inside:

```
experiments/
```

- protonet_best_<timestamp>.pt  
- protonet_last_<timestamp>.pt  

---

### Run Episode Inference

After training:

macOS / Linux:

```bash
PYTHONPATH=src python scripts/infer_episode.py --ckpt experiments/protonet_best_<timestamp>.pt
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts/infer_episode.py --ckpt experiments/protonet_best_<timestamp>.pt
```

This will:
- Load the trained model  
- Sample a new few-shot episode  
- Compute prototypes  
- Classify query samples  
- Print predicted vs true labels  

---

## Notes

- The dataset must be organized inside `data/train` and optionally `data/test`.
- Each class folder must contain at least `k_shot + q_query` images.
- Default setup is 4-way 5-shot episodic training.

---
