from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
import numpy as np
import matplotlib.pyplot as plt

def naive_bayes_function(X_train, X_test, y_train, y_test):
    # Initialize Gaussian Naive Bayes classifier
    nb_clf = GaussianNB()

    # Fit the model on the training data
    nb_clf.fit(X_train, y_train)

    # Predict probabilities for train and test sets
    train_probs = nb_clf.predict_proba(X_train)
    test_probs = nb_clf.predict_proba(X_test)

    # Calculate log loss for train and test sets
    train_log_loss = log_loss(y_train, train_probs, labels=nb_clf.classes_)
    test_log_loss = log_loss(y_test, test_probs, labels=nb_clf.classes_)

    # Display log loss for train and test sets
    print(f'Train Log Loss: {train_log_loss:.5f}')
    print(f'Test Log Loss: {test_log_loss:.5f}')

    # Predict labels for the test set
    predicted_labels = nb_clf.predict(X_test)

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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_clf.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, predicted_labels))

# Example usage:
# naive_bayes_function(X_train, X_test, y_train, y_test)
