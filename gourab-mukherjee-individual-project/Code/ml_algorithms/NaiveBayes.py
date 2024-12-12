from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import (
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
    cohen_kappa_score
)
import matplotlib.pyplot as plt

def naive_bayes_function(X_train, X_test, y_train, y_test):
    # Initialize Gaussian Naive Bayes classifier
    # Ensure y_train and y_test are 1D arrays
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

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
    # Calculate F1 Scores
    f1_macro = f1_score(y_test, predicted_labels, average='macro')
    f1_micro = f1_score(y_test, predicted_labels, average='micro')
    print(f'F1 Macro: {f1_macro:.5f}')
    print(f'F1 Micro: {f1_micro:.5f}')

    # Calculate AUC
    try:
        test_auc = roc_auc_score(y_test, test_probs[:, 1])
        print(f'Test AUC: {test_auc:.5f}')
    except ValueError:
        print("AUC calculation failed. Ensure y_test and test_probs are binary.")

    # Calculate Cohen's Kappa
    cohen_kappa = cohen_kappa_score(y_test, predicted_labels)
    print(f'Cohen Kappa: {cohen_kappa:.5f}')


    # Display confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_clf.classes_)
    disp.plot(cmap='Blues', values_format='d')

    plt.title('Confusion Matrix')
    plt.show()
