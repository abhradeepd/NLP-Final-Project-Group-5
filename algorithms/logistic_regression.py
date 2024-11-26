from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def logistic_regression_function(X_train, X_test, y_train, y_test, max_iter=1000, random_state=42):
    """
    Logistic Regression Training and Evaluation Function

    Parameters:
    - X_train: Training feature set (DataFrame or array).
    - X_test: Test feature set (DataFrame or array).
    - y_train: Training labels.
    - y_test: Test labels.
    - max_iter: Maximum iterations for logistic regression optimization.
    - random_state: Random state for reproducibility.

    Outputs:
    - Log loss for training and test sets.
    - F1 scores (micro, macro, weighted) for the test set.
    - Confusion matrix with counts and percentages.
    """
    # Initialize Logistic Regression model
    logreg_model = LogisticRegression(max_iter=max_iter, random_state=random_state)

    # Fit the model to the training data
    logreg_model.fit(X_train, y_train)

    # Predict probabilities for train and test sets
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

    # Calculate F1 scores for test data
    f1_micro = f1_score(y_test, predicted_labels, average="micro")
    f1_macro = f1_score(y_test, predicted_labels, average="macro")
    f1_weighted = f1_score(y_test, predicted_labels, average="weighted")

    print("\nF1 Scores on Test Data:")
    print(f"F1 Micro: {f1_micro:.5f}")
    print(f"F1 Macro: {f1_macro:.5f}")
    print(f"F1 Weighted: {f1_weighted:.5f}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_labels))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)

    # Normalize confusion matrix for percentages
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Count-based confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg_model.classes_)
    disp.plot(ax=ax[0], cmap="Blues", values_format="d")
    ax[0].set_title("Confusion Matrix (Counts)")

    # Percentage-based confusion matrix
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=logreg_model.classes_,
        yticklabels=logreg_model.classes_,
        ax=ax[1],
    )
    ax[1].set_title("Confusion Matrix (Percentages)")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

    # Training summary
    print("\nTraining Summary:")
    print(f"Number of Training Samples: {X_train.shape[0]}")
    print(f"Number of Testing Samples: {X_test.shape[0]}")
    print(f"Number of Features: {X_train.shape[1]}")