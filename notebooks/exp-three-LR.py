import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Set MLflow Tracking URI & DAGsHub integration
MLFLOW_TRACKING_URI = "https://dagshub.com/bhauryal7/Multiclass-Text-Classification.mlflow"
dagshub.init(repo_owner="bhauryal7", repo_name="Multiclass-Text-Classification", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("LR Hyperparameter Tuning")


# ==========================
# Text Preprocessing Functions
# ==========================
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
def clean_text(text):
    # Lowercasing
    text = text.lower()
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Removing stop words
    stop_words = set(stopwords.words("german"))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Removing numbers
    tokens = [word for word in tokens if not word.isdigit()]
    # Removing extra whitespaces
    text = ' '.join(tokens)
    return text

def normalize_text(df):
    try:
        df['DESCRIPTION']= df['DESCRIPTION'].apply(clean_text)
        df['Herstellung']=df['Herstellung'].str.strip().str.lower()
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise


# ==========================
# Load & Prepare Data
# ==========================
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=20000)

label_encoder = LabelEncoder()
def load_and_prepare_data(filepath):
    """Loads, preprocesses, and vectorizes the dataset."""
    df = pd.read_csv(filepath)
    df = normalize_text(df)
    df["Herstellung"] = label_encoder.fit_transform(df["Herstellung"])
    X = vectorizer.fit_transform(df["DESCRIPTION"])
    y = df["Herstellung"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    
    param_grid = {
         "C": [ 10],
        # "solver":['lbfgs','liblinear','saga']
    }
    
    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression(penalty='l2',class_weight='balanced',max_iter=1000,solver='liblinear'), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(grid_search.cv_results_["params"], 
                                                 grid_search.cv_results_["mean_test_score"], 
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred,average='weighted'),
                    "recall": recall_score(y_test, y_pred,average='weighted'),
                    "f1_score": f1_score(y_test, y_pred,average='weighted'),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }
                
                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_accuracy = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_accuracy )
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"\nBest Params: {best_params} | Best Accuracy: {best_accuracy:.4f}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test) = load_and_prepare_data("notebooks/data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test)