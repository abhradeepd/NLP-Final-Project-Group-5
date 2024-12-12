from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
import numpy as np
import matplotlib.pyplot as plt

def sgd_random_search_v1(X_train, X_test, y_train, y_test):
    # Hyperparameter grid for RandomizedSearchCV
    param_dist = {
        'alpha': [10 ** x for x in range(-4, 1)],  # Smaller range of values
        'penalty': ['elasticnet'],
        'loss': ['log_loss'],  # Correct loss for SGDClassifier
        'l1_ratio': [0.15, 0.5]  # Focus on fewer values for l1_ratio
    }

    # Initialize SGDClassifier
    sgd_clf = SGDClassifier(random_state=42)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        sgd_clf,
        param_distributions=param_dist,
        scoring='neg_log_loss',
        n_iter=10,  # Number of iterations for randomized search
        cv=3,
        n_jobs=1,
        refit=True  # Ensure the best model is refit
    )

    # Perform RandomizedSearchCV on the data
    random_search.fit(X_train, y_train)

    # Best model calibration
    best_sgd_clf = random_search.best_estimator_
    calibrated_clf = CalibratedClassifierCV(best_sgd_clf, method="sigmoid")
    calibrated_clf.fit(X_train, y_train)

    # Print the best parameters
    print("Best Parameters:", random_search.best_params_)

    # Predict probabilities for train and test sets
    train_probs = calibrated_clf.predict_proba(X_train)
    test_probs = calibrated_clf.predict_proba(X_test)

    # Calculate log loss for train and test sets
    train_log_loss = log_loss(y_train, train_probs)
    test_log_loss = log_loss(y_test, test_probs)

    # Display log loss for train and test sets
    print(f'Train Log Loss: {train_log_loss:.5f}')
    print(f'Test Log Loss: {test_log_loss:.5f}')

    # Predict labels for the test set
    predicted_labels = calibrated_clf.predict(X_test)

    # Calculate F1 scores
    macro_f1 = f1_score(y_test, predicted_labels, average='macro')
    micro_f1 = f1_score(y_test, predicted_labels, average='micro')
    weighted_f1 = f1_score(y_test, predicted_labels, average='weighted')

    # Display F1 scores
    print(f'Macro F1 Score: {macro_f1:.5f}')
    print(f'Micro F1 Score: {micro_f1:.5f}')
    print(f'Weighted F1 Score: {weighted_f1:.5f}')

    # Display confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=calibrated_clf.classes_)
    disp.plot(cmap='Blues', values_format='d')

    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, predicted_labels))

    return {
        "best_params": random_search.best_params_,
        "train_log_loss": train_log_loss,
        "test_log_loss": test_log_loss,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
    }