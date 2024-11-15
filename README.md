# Multiclass Text Classification with Machine Learning Models

This repository contains python code in two jupyter notebooks for classifying text data using various machine (and deep) learning models. The goal is to automate the process of item categorization.
 
# Requirements:

* NumPy
* Pandas
* NLTK
* scikit-learn
* Tensorflow

# Dataset:
The dataset contains a list of around 42,000 food and non-food items, categorized into more than 300 labels. The excel file contains the data and the list of items is in the column 'DESCRIPTION', which is in German, while the corresponding categories are stored in the column 'Herstellung'. The dataset used in this project is provided by Carbotech.

# Results:
The results of each model on the dataset are as follows:

|  Model | Accuracy |
|----------|----------|
| Logistic Regression | 80.52% |
| Support Vector Machine | 79.73% |
| Multinomial Naive Bayes | 78.25% |
| Stochastic Gradient Descent | 79.11% |
| Randomforest | 73.7% |
| XGBoost | 65.78 |
| RNN | 62.3% |
| GRU | 62.3% |
| LSTM | 72% |

# Future Improvements

- Experiment with a BERT-based model for potentially better perfromance.
