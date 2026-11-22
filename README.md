# breast-cancer-classification-ml
Dự án ứng dụng Machine Learning để phân loại khối u vú lành tính và ác tính, so sánh hiệu quả của 5 mô hình và tối ưu hóa tham số

# Breast Cancer Classification using Machine Learning

## 1. Project Overview

This project applies Machine Learning algorithms
to classify breast tumors as Benign (B) or
Malignant (M) using the Breast Cancer Wisconsin
(Diagnostic) dataset.

The main objective is to compare different
classification models, evaluate their performance,
and improve selected models through
hyperparameter tuning.

This is an academic Machine Learning project.

## 2. Dataset

- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository
- Number of observations: 569
- Number of input features: 30
- Target variable: Diagnosis
- Classes:
  - B: Benign
  - M: Malignant

Dataset source:
https://archive.ics.uci.edu/dataset/17/
breast+cancer+wisconsin+diagnostic

## 3. Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- UCI ML Repository

## 4. Project Workflow

### 4.1 Exploratory Data Analysis (EDA)

- Dataset overview
- Descriptive statistics
- Feature distribution
- Outlier detection
- Correlation analysis

### 4.2 Data Preprocessing

- Duplicate checking
- Missing value checking
- Data validation
- Train-test split (80:20)
- Feature scaling using StandardScaler

### 4.3 Model Development

Five classification algorithms were evaluated:

1. Gaussian Naive Bayes
2. Logistic Regression
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Random Forest

### 4.4 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

### 4.5 Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation
was applied to:

- Support Vector Machine
- Random Forest

## 5. Model Performance

The following results are from the current
project report and should be updated after
the final code and evaluation are reconciled.

| Model | Accuracy |
|---|---:|
| Gaussian Naive Bayes | 92.11%* |
| Logistic Regression | 96.49% |
| KNN (K=5) | 95.61% |
| Random Forest (Tuned) | 97.37% |
| SVM (Tuned) | 98.25% |

*The Gaussian Naive Bayes result requires
reconciliation with the original conclusion table.

SVM achieved the highest reported test
accuracy of 98.25%.

## 6. Key Findings

- SVM achieved the best reported overall
  classification performance after tuning.
- Random Forest also achieved strong
  classification results.
- Feature scaling was important for
  distance-based and margin-based models.
- Hyperparameter tuning improved the
  reported SVM performance.

## 7. Repository Structure

- src/: Python implementation
- notebooks/: Exploratory analysis and modeling
- reports/: Academic project report
- results/: Evaluation metrics and visualizations
- data/: Dataset source information

## 8. How to Run

Clone this repository:

git clone YOUR_REPOSITORY_URL

Install dependencies:

pip install -r requirements.txt

Run the Python script:

python src/breast_cancer_classification.py

Note: The dataset is retrieved from UCI
using the ucimlrepo package, so internet
access is required.

## 9. Limitations

- Evaluation is based on a single train-test split.
- Performance should be validated on independent
  external datasets.
- The project is an academic experiment and
  is not intended for clinical diagnosis.

## 10. Author

Thuy Linh Le

University of Economics and Law
Vietnam National University HCMC

Major: Applied Mathematics in Economics,
Management and Finance
