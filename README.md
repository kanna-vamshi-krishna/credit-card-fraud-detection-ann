# Credit Card Fraud Detection using ANN (Deep Learning)

This project detects fraudulent credit card transactions using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras**.  
Since fraud datasets are highly imbalanced, the project focuses on **AUC, Precision, Recall, PR-AUC** rather than only Accuracy.

---

## Problem Statement
Credit card fraud causes huge financial losses every year. The goal of this project is to build a deep learning model that can classify a transaction as:
- **0 → Genuine Transaction**
- **1 → Fraud Transaction**

---

## Dataset
- Source: Kaggle - Credit Card Fraud Detection (mlg-ulb dataset)
- File: `creditcard.csv`
- Target Column: `Class` (0 = Normal, 1 = Fraud)

⚠️ The dataset is **highly imbalanced**, so accuracy alone is not a good metric.

Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Project Workflow
1. Load dataset and perform basic checks  
2. Split data into **Train / Validation / Test** using stratification  
3. Apply **StandardScaler** for feature scaling  
4. Handle class imbalance using **Class Weights**  
5. Build ANN model using TensorFlow/Keras  
6. Train the model with **Early Stopping**  
7. Predict fraud probability and generate **y_pred**  
8. Tune classification threshold (not fixed 0.5)  
9. Evaluate using:
   - Confusion Matrix
   - Classification Report
   - ROC-AUC
   - PR-AUC (Average Precision)
   - ROC Curve
   - Precision-Recall Curve  
10. Save trained model and scaler

---

## Model Used
Artificial Neural Network (ANN):
- Dense Layers with ReLU activation
- Dropout layers to reduce overfitting
- L2 Regularization
- Output layer: Sigmoid (Binary Classification)

---

## Evaluation Metrics
Because fraud detection is an imbalanced problem, the main metrics used are:
- **ROC-AUC**
- **PR-AUC (Average Precision)**
- **Precision**
- **Recall**
- **F1-Score**

---

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

---

## How to Run This Project

### 1) Install dependencies
```bash
pip install -r requirements.txt

