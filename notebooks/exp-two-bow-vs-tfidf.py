import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import scipy.sparse

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==================================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/bhauryal7/Multiclass-Text-Classification.mlflow",
    "dagshub_repo_owner": "bhauryal7",
    "dagshub_repo_name": "Multiclass-Text-Classification",
    "experiment_name": "BOW vs TFIDF"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ===============================
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

# ========================== LOAD & PREPROCESS DATA ==========================
label_encoder = LabelEncoder()
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df["Herstellung"] = label_encoder.fit_transform(df["Herstellung"])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ========================== FEATURE ENGINEERING ==========================
VECTORIZERS = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    #'LogisticRegression': LogisticRegression(),
    #'MultinomialNB': MultinomialNB(),
    #'XGBoost': XGBClassifier(),
    #'RandomForest': RandomForestClassifier(),
    #'GradientBoosting': GradientBoostingClassifier(),
    'SupportVectorMachine': SVC(),
    'StochasticGradientDescent': SGDClassifier()
}

# ========================== TRAIN & EVALUATE MODELS ==========================
def train_and_evaluate(df):
    with mlflow.start_run(run_name="All Experiments") as parent_run:
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                    try:
                        # Feature extraction
                        X = vectorizer.fit_transform(df['DESCRIPTION'])
                        y = df['Herstellung']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

                        # Log preprocessing parameters
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })

                        # Train model
                        model = algorithm
                        model.fit(X_train, y_train)

                        # Log model parameters
                        log_model_params(algo_name, model)

                        # Evaluate model
                        y_pred = model.predict(X_test)
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred,average='weighted'),
                            "recall": recall_score(y_test, y_pred,average='weighted'),
                            "f1_score": f1_score(y_test, y_pred,average='weighted')
                        }
                        mlflow.log_metrics(metrics)

                        # Log model
                        # mlflow.sklearn.log_model(model, "model")
                        input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                        mlflow.sklearn.log_model(model, "model", input_example=input_example)

                        # Print results for verification
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

                    except Exception as e:
                        print(f"Error in training {algo_name} with {vec_name}: {e}")
                        mlflow.log_param("error", str(e))

def log_model_params(algo_name, model):
    """Logs hyperparameters of the trained model to MLflow."""
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log["C"] = model.C
    elif algo_name == 'MultinomialNB':
        params_to_log["alpha"] = model.alpha
    elif algo_name == 'XGBoost':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'SupportVectorMachine':
        params_to_log['C'] = model.C
    elif algo_name =='StochasticGradientDescent':
        params_to_log['alpha'] = model.alpha        



    mlflow.log_params(params_to_log)

# ========================== EXECUTION ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)