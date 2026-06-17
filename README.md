# Titanic Survival Prediction

COGS 109 Final Project – Spring 2026

## Overview
This project predicts passenger survival on the Titanic using demographic and ticket information. We compare Logistic Regression and K‑Nearest Neighbors using 5‑fold cross‑validation, regularization, PCA, and bootstrap resampling.

## Key Results
- Best model: KNN (k=21) – **78.8%** test accuracy
- Logistic regression – **79.3%** test accuracy
- Gender and passenger class are the strongest predictors
- Bootstrap 95% CI for KNN accuracy: [72.6%, 84.4%]

## Repository Contents
- `train.csv` – dataset
- `titanic_project.ipynb` – full Jupyter notebook
- `Report_Cogs109_Titanic.pdf` – final written report
- `plots/` – figures (CV error, confusion matrix, ROC curve, survival bar plot)

## Methods
- Data cleaning (median imputation, one‑hot encoding)
- Standardization for KNN
- 5‑fold cross‑validation for model selection
- Regularization (L1/L2)
- PCA dimensionality reduction
- Bootstrap confidence intervals
- Evaluation: accuracy, confusion matrix, ROC/AUC

## Dependencies
Python 3.x, pandas, numpy, scikit‑learn, matplotlib, seaborn
