from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
    cohen_kappa_score
)
import matplotlib.pyplot as plt

def random_forest_random_search(X_train, X_test, y_train, y_test):
    # Ensure y_train and y_test are 1D arrays
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Hyperparameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Initialize Random Forest classifier
    rf_clf = RandomForestClassifier(random_state=42)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf_clf,
        param_distributions=param_dist,
        scoring='neg_log_loss',
        n_iter=5,  # Adjust for more iterations if needed
        cv=3,
        n_jobs=-1,
        verbose=2  # Adjust verbosity level
    )

    # Perform RandomizedSearchCV on the data
    random_search.fit(X_train, y_train)

    # Print the best parameters
    print("Best Parameters:", random_search.best_params_)

    # Get the best model from RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Predict probabilities for train and test sets
    train_probs = best_model.predict_proba(X_train)
    test_probs = best_model.predict_proba(X_test)

    # Calculate log loss for train and test sets
    train_log_loss = log_loss(y_train, train_probs, labels=best_model.classes_)
    test_log_loss = log_loss(y_test, test_probs, labels=best_model.classes_)

    # Display log loss for train and test sets
    print(f'Train Log Loss: {train_log_loss:.5f}')
    print(f'Test Log Loss: {test_log_loss:.5f}')

    # Predict labels for the test set
    predicted_labels = best_model.predict(X_test)

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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap='Blues', values_format='d')

    plt.title('Confusion Matrix')
    plt.show()

# Example usage:
# random_forest_random_search(X_train, X_test, y_train, y_test)
