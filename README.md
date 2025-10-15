# 🌶 Pepper Disease Detection with EfficientNet-B0

This project implements a **deep learning pipeline for automatic disease detection in bell pepper leaves**, using the **EfficientNet-B0** architecture trained in **PyTorch / FastAI**.  
The system classifies images as either **Healthy** or **Bacterial Spot** with **99.6% accuracy**.

---

## 🚀 Overview

- **Architecture:** EfficientNet-B0  
- **Frameworks:** PyTorch + FastAI + TIMM  
- **Dataset:** Local ZIP (custom dataset with `train/valid` folders)  
- **Precision:** Mixed FP16 for efficient GPU training  
- **Accuracy:** 99.6% (EXCELLENT level)  

---

## 🧠 Training Methodology

The model was trained in **two phases**:

### **Phase 1 — Frozen Training**
- Trained only the classification head while keeping the EfficientNet backbone frozen.
- Duration: 5 epochs
- Learning rate: `2.09e-03`
- Achieved ~99.2% accuracy early on.

### **Phase 2 — Fine-tuning (Unfrozen)**
- Entire network unfrozen for full fine-tuning.
- Duration: 15 epochs
- Discriminative learning rates from `1.10e-05` → `1.10e-04`
- Achieved **final accuracy of 99.6%**, with improved recall and F1-score.

---

## 📊 Dataset Summary

| Split | Class | Images |
|:------|:------|-------:|
| **Train** | Bell pepper Bacterial spot | 3,449 |
| **Train** | Bell pepper Healthy | 4,014 |
| **Valid** | Bell pepper Bacterial spot | 1,031 |
| **Valid** | Bell pepper Healthy | 1,488 |
| **Total** | 9,982 |


---

## Dataset Examples
<!-- imagen 1: ejemplo de batch del dataset -->
<img width="1800" height="1771" alt="batch_predictions_batch_1" src="https://github.com/user-attachments/assets/607c4378-4f67-4969-b373-dcf341a3c3c2" />

Example training batch showing augmentations and class balance.

---

## 📈 Training Performance

| Phase | Accuracy | Precision | Recall | F1-score |
|:------|----------:|-----------:|--------:|----------:|
| Frozen | 0.992 | 0.991 | 0.996 | 0.994 |
| Fine-tuned | **0.996** | **0.995** | **0.998** | **0.996** |

---

## 🔍 Evaluation Results

### 1️⃣ Confusion Matrix
<img width="1000" height="1000" alt="confusion_matrix" src="https://github.com/user-attachments/assets/601b3ae2-8e13-44ed-9723-d5e4eb423297" />

Shows excellent separation between “Healthy” and “Bacterial Spot”.

---

### 2️⃣ Hardest Cases
<!-- imagen 3: top losses -->
<img width="4018" height="2670" alt="top_losses" src="https://github.com/user-attachments/assets/5d9b1b92-4302-4888-9836-c7cd26def029" />

Visualizes samples the model found most challenging.

---

### 3️⃣ Final Predictions
<img width="3468" height="2488" alt="final_predictions" src="https://github.com/user-attachments/assets/41e15ccd-44f1-42e1-9440-e24d217c6b60" />


Examples of correctly and incorrectly classified leaves.



