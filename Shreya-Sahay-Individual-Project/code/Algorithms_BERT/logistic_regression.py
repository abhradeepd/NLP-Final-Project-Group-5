from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
    accuracy_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import vstack
import matplotlib.pyplot as plt
import numpy as np
import time


def efficient_logistic_regression_function(
        X_train, X_test, y_train, y_test,
        max_iter=1000, random_state=42,
        penalty="l2", C=1.0, solver="lbfgs"
):
    """
    Efficient Logistic Regression Training and Evaluation Function for Large Datasets.

    Parameters:
    - X_train: Training feature set (TfidfVectorizer output or sparse array).
    - X_test: Test feature set (TfidfVectorizer output or sparse array).
    - y_train: Training labels (array or series).
    - y_test: Test labels (array or series).
    - max_iter: Maximum iterations for logistic regression optimization.
    - random_state: Random state for reproducibility.
    - penalty: Regularization type ("l1", "l2", or "elasticnet").
    - C: Inverse of regularization strength (smaller -> stronger regularization).
    - solver: Solver for optimization ("saga" recommended for large datasets).

    Outputs:
    - Log loss, F1 scores, and accuracy for the test set.
    """

    # Start timer
    start_time = time.time()

    print("Initializing Logistic Regression...")

    # Initialize Logistic Regression model
    logreg_model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        penalty=penalty,
        C=C,
        solver=solver,
    )

    # Fit the model to the training data
    print("Training the model...")
    logreg_model.fit(X_train, y_train)

    # Training time
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")

    # Predict probabilities for train and test sets
    print("Making predictions...")
    train_probs = logreg_model.predict_proba(X_train)
    test_probs = logreg_model.predict_proba(X_test)

    # Calculate log loss for train and test sets
    train_log_loss = log_loss(y_train, train_probs)
    test_log_loss = log_loss(y_test, test_probs)

    # Display log loss for train and test sets
    print(f"Train Log Loss: {train_log_loss:.5f}")
    print(f"Test Log Loss: {test_log_loss:.5f}")

    # Predict labels for the test set
    predicted_labels = logreg_model.predict(X_test)

    # Calculate accuracy and F1 scores
    accuracy = accuracy_score(y_test, predicted_labels)
    f1_micro = f1_score(y_test, predicted_labels, average="micro")
    f1_macro = f1_score(y_test, predicted_labels, average="macro")
    f1_weighted = f1_score(y_test, predicted_labels, average="weighted")

    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.5f}")
    print(f"F1 Micro: {f1_micro:.5f}")
    print(f"F1 Macro: {f1_macro:.5f}")
    print(f"F1 Weighted: {f1_weighted:.5f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)

    # Visualize confusion matrix (counts only)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg_model.classes_)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Counts)")
    plt.show()

    # Training summary
    print("\nTraining Summary:")
    print(f"Number of Training Samples: {X_train.shape[0]}")
    print(f"Number of Testing Samples: {X_test.shape[0]}")
    print(f"Number of Features: {X_train.shape[1]}")
