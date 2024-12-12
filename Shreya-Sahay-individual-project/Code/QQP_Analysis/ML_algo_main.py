import torch
import numpy as np
import time
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import xgboost as xgb
import os
from sklearn.naive_bayes import GaussianNB
# Configure logging
logging.basicConfig(filename="results_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")


def create_pipeline(algorithm):
    """Create a pipeline based on the selected algorithm."""
    if algorithm == "random_forest":
        return Pipeline([
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=None,  # No restriction on depth
                min_samples_split=2,  # Minimum samples to split is 2
                min_samples_leaf=1,  # Minimum samples in a leaf is 1
                random_state=42
            ))
        ])
    elif algorithm == "logistic_regression":
        return Pipeline([
            ('scaler', StandardScaler()),  # Scaling required for Logistic Regression
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
    elif algorithm == "svm":
        return Pipeline([
            ('scaler', StandardScaler()),  # Scaling required for SVM
            ('classifier', SVC(kernel='rbf', probability=True, random_state=42))  # RBF kernel added
        ])
    elif algorithm == "xgboost":
        return Pipeline([
            ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
    elif algorithm == "naive_bayes":
        return Pipeline([
            ('scaler', StandardScaler()),  # Standardization for GaussianNB
            ('classifier', GaussianNB())  # Gaussian Naive Bayes classifier
        ])
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def load_and_prepare_data(data_path):
    """Load and prepare data."""
    logging.info("Loading data...")
    try:
        data = torch.load(data_path)

        # Extract data components
        ids = data['id'].numpy()
        q1_feats = data['q1_feats_bert'].numpy()
        q2_feats = data['q2_feats_bert'].numpy()
        features = data['features'].numpy()
        labels = data['labels'].numpy()

        combined_features = np.hstack((q1_feats, q2_feats, features))
        logging.info(f"Data loaded successfully. Combined features shape: {combined_features.shape}")
        return combined_features, labels, ids
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def split_data(combined_features, labels, ids):
    """Split data into train, validation, and test sets."""
    logging.info("Splitting data...")
    train_idx, temp_idx = train_test_split(range(len(ids)), test_size=0.15, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=2 / 3, random_state=42)

    def split(indices):
        return combined_features[indices], labels[indices]

    X_train, y_train = split(train_idx)
    X_val, y_val = split(val_idx)
    X_test, y_test = split(test_idx)

    logging.info("Data split completed.")
    logging.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model, X, y, stage="Validation"):
    """Evaluate the model and return metrics."""
    logging.info(f"Evaluating on {stage} data...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro"),
        "f1_micro": f1_score(y, y_pred, average="micro"),
        "auc": roc_auc_score(y, y_proba)
    }
    logging.info(f"{stage} Metrics: {metrics}")
    return metrics


def main(algorithm="random_forest"):
    start_time = time.time()
    data_path = "../Data/merged_features_embeddings.pt"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        exit(1)

    file_size = os.path.getsize(data_path) / (1024 * 1024)  # Convert to MB
    print(f"File size: {file_size:.2f} MB")

    try:
        # Load and prepare data
        combined_features, labels, ids = load_and_prepare_data(data_path)

        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(combined_features, labels, ids)

        # Create pipeline
        pipeline = create_pipeline(algorithm)
        logging.info(f"Pipeline created for algorithm: {algorithm}")

        # Train the model
        logging.info("Starting training...")
        train_start = time.time()
        pipeline.fit(X_train, y_train)
        logging.info(f"Training completed in {time.time() - train_start:.2f} seconds.")

        # Evaluate on validation data
        evaluate_model(pipeline, X_val, y_val, stage="Validation")

        # Evaluate on test data
        evaluate_model(pipeline, X_test, y_test, stage="Test")

        # Log total time
        logging.info(f"Total time taken: {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    #main(algorithm="random_forest")
    #main(algorithm="svm")
    main(algorithm="naive_bayes")
