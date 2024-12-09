from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
import numpy as np
def splitting(df, y_true,test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(df, y_true, stratify=y_true, test_size=0.3)
    # Convert lists to DataFrames if they are not already
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    # Create a DataFrame to display the sizes of the splits
    split_sizes = pd.DataFrame({
        'Data Split': ['X_train', 'X_test', 'y_train', 'y_test'],
        'Size': [X_train.shape[0], X_test.shape[0], y_train.shape[0], y_test.shape[0]]
    })

    # Display the split sizes in tabular format
    print("Size of Data Splits:")
    print(split_sizes)

    # Print head(5) for each split
    print("Head of X_train:")
    print(X_train.head())

    print("\nHead of X_test:")
    print(X_test.head())

    print("\nHead of y_train:")
    print(y_train.head())

    print("\nHead of y_test:")
    print(y_test.head())

    return X_train, X_test, y_train, y_test



def distribution_outputvariable_train_test(y_train,y_test):
    # Plotting the distribution of the output variable in train data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train.iloc[:, 0])
    plt.title('Distribution of Output Variable in Train Data')

    # Plotting the distribution of the output variable in test data
    plt.subplot(1, 2, 2)
    sns.countplot(x=y_test.iloc[:, 0])
    plt.title('Distribution of Output Variable in Test Data')

    plt.show()

def random_model(y_train,y_test):
    np.random.seed(42)  # Set seed for reproducibility
    random_predictions_train = np.random.rand(len(y_train))

    # Ensure the predictions sum up to 1 for each sample
    random_predictions_train /= random_predictions_train.sum(keepdims=True)

    # Calculate log loss for y_train
    log_loss_train = log_loss(y_train, random_predictions_train)

    # Display the log loss for the training data
    print(f'Log Loss for Training Data: {log_loss_train:.5f}')

    # Generate random predictions for y_test
    random_predictions_test = np.random.rand(len(y_test))
    random_predictions_test /= random_predictions_test.sum(keepdims=True)

    # Calculate log loss for y_test
    log_loss_test = log_loss(y_test, random_predictions_test)

    # Display the log loss for the test data
    print(f'Log Loss for Test Data: {log_loss_test:.5f}')





