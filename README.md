# Thyroi Disorders Classification & Segmentation

This repository contains machine learning and deep learning pipelines for the analysis of thyroid disorders, including ultrasound image segmentation, thyroid nodule classification, and clinical disease classification for Hashimoto’s thyroiditis and Graves’ disease.

## 📚 Publication

This repository is associated with the following peer-reviewed publication:

**Title:**  
“A Machine Learning and Deep Learning Approach for the Classification of Thyroid Disorders Using Multi-Source Clinical Data”

**Authors:**  
Kypros Andreou, Marios G. Krokidis, Themis P. Exarchos, Panagiotis Vlamos, Eleftherios Georgakopoulos, Costas Toufexis, Nikos L. Papaloizou.

**Source:**  
BioMedInformatics, MDPI, 2026  

**Link:**  
https://www.mdpi.com/3915474
  
## 📁 Repository Structure
Thyroid_Disorders_Classification/
│
├── Thyroid_Ultrasound_Classification/
├── Ultrasound_Nodule_Segmentation/
├── Hashimoto_Classification/
└── Graves_Classification/

## 🧠 Project Overview

This project is divided into four main components:

### 1. Thyroid Ultrasound Classification (CNN)
This module focuses on classification of thyroid nodules from ultrasound images.

**Tasks:**
- Benign vs malignant nodule classification
- CNN-based feature learning ( EfficientNet-B0)
- Model evaluation and performance analysis

**Outputs:**
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrix analysis

---

### 2. Ultrasound_Nodule_Segmentation (U-Net)
This module performs thyroid nodule segmentation from ultrasound images.

**Model:**
- U-Net architecture

**Tasks:**
- Pixel-wise segmentation of thyroid nodules

**Outputs:**
- Segmentation masks
- Visualization of predicted vs ground truth masks

---

### 3. Hashimoto Classification (Machine Learning)
This module focuses on classification of Hashimoto’s thyroiditis using structured clinical data.

**Tasks:**
- Hashimoto vs non-Hashimoto classification
- Feature-based machine learning pipeline

**Typical models:**
- XGBoost
- Random Forest
- Gradient Boosting
- Decision Tree

---

### 4. Graves Classification (Machine Learning)
This module focuses on Graves’ disease classification using clinical or laboratory features.

**Tasks:**
- Graves vs non-Graves classification
- Classical machine learning models on tabular data

**Typical models:**
- XGBoost
- Random Forest
- Gradient Boosting
- Decision Tree
---

## 📊 Evaluation Metrics

Across all modules, performance is evaluated using:

- Accuracy
- Precision
- Recall
- Roc-Auc
- F1-score
- Confusion Matrix

---
