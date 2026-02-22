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
