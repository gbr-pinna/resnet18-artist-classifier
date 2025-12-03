# Artist Recognition — ResNet18 (PyTorch Demo)

This repository contains a small **pedagogical exercise** demonstrating how to train and evaluate a convolutional neural network (ResNet-18) with **PyTorch** for the task of **painting artist recognition**.

The goal of the project is purely educational: to practice transfer learning, dataset preparation, data augmentation, and model evaluation — not to produce a production-ready classifier.

---

## Repository Contents

- **`artist_recognition_training.ipynb`**  
  Notebook used to prepare the dataset, apply data augmentation, train a pretrained **ResNet-18** model on the **top 3 painters** from the Kaggle dataset, and save the best model weights.

- **`artist_recognition_testing.ipynb`**  
  Notebook used to load the trained model and evaluate it on a small manually collected test set of unseen paintings.

- **`test_set/`**  
  Folder containing the test images used for evaluation (13 paintings per artist).

- **`top3_painters/`**  
  Folder containing the filtered training images for the three most represented painters from the Kaggle dataset.

- **`best_painter_top3_resnet18.pth`**  
  Saved weights of the trained ResNet-18 model used during testing.

---

## Dataset Source

Training images are taken from the Kaggle dataset:

**Best Artworks of All Time**  
https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time

Due to heavy class imbalance, only the **top three most represented painters** were used for training.

---

## Purpose

This project is meant as a **learning exercise** to demonstrate:

- Using pretrained CNNs (ResNet-18) with PyTorch.
- Freezing and fine-tuning layers for transfer learning.
- Data preprocessing and augmentation.
- Training/validation workflows.
- Simple evaluation on a held-out test set.

It is **not intended** as a real-world art authentication or production ML system.

---

All experiments are intended to be run from the provided Jupyter notebooks.
