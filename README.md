# Multiclass Text Classification for German Food Product Descriptions

This project focuses on building a multiclass text calssification model using traditional machine learning techniques to classify food product descriptions into multiple predefined categories. 

# Project Overview

- **Input:** Short product descriptions in German (average 6-7 words)
- **Dataset size** ~42k product descriptions
- **Output:** Predicted category label from multiple classes (~350)
- **Tech Stack:** Python, Scikit-learn, Numpy, Pandas, Flask, Docker, GitHub Actions


# Features

- Preprocessing of text data
- TF-IDF feature extraction
- Model training using classical ML models (e.g., Logistic Regression, Naive Bayes)
- Performance evaluation (accuracy, precision, recall)
- MLFlow for data versioning and experiment tracking
- DVC (Data Version Control) for managing datasets and ML pipeline stages
- Simple Flask-based API for serving predictions
- Docker for containerizing the application
- CI/CD pipeline using GitHub Actions for automated testing and deployment to AWS ECR


## Analysis & Techniques

To train robust models on this imbalanced dataset, the following steps were taken:

- **TF-IDF Vectorization:** Used `TfidfVectorizer` to extract informative features from short texts (average 5–6 words per row), capturing term importance while reducing noise from common words
- **Class Weight Handling:** Utilised  the `class_weight='balanced'` argument in models to counter the imbalance across ~350 categories
- **Model Comparison:** Several models were trained and compared, e.g., Logistic Regression, SVM, SGD, Random Forest, Naive Bayes
- **Experiment Tracking:** Each experiment (model, vectorizer config, metrics) was logged using MLflow for reproducibility and analysis

---

# Future Improvements
- Try using word embeddings like Word2Vec to capture similarities between words, even if they are not exactly the same (e.g., “organic” and “bio”). Also explore models like BERT, which understand word meaning based on context and can handle rare or unseen words better. These methods go beyond TF-IDF and simple models by using dense, meaningful representations instead of sparse word counts, and can improve performance on more complex or varied product descriptions.

# Demo

Check out the live demo: [Item category predictor](https://multiclass-text-classification.onrender.com/)

![Flask App Screenshot](./images/flask_app.png) 
