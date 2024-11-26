import pandas as pd
import numpy as np
import torch
from algorithms.logistic_regression import logistic_regression_function
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
#
# Debugging Function
def debug_print(message, data=None):
    """Helper function to print debugging information."""
    print(f"DEBUG: {message}")
    if data is not None:
        print(data)

# Load train.csv and get the number of rows
# try:
#     df = pd.read_csv('Data/train.csv')
#     nrows = len(df)
#     debug_print("Number of rows in train.csv", nrows)
# except Exception as e:
#     debug_print("Failed to load train.csv", e)
#     raise

# Load bert_embeddings.pt
try:
    embeddings_data = torch.load('Data/bert_embeddings.pt')
    debug_print("Loaded bert_embeddings.pt successfully", {
        "Keys in .pt file": list(embeddings_data.keys())
    })
except Exception as e:
    debug_print("Failed to load bert_embeddings.pt", e)
    raise

# Extract data from the .pt file
try:
    ids = embeddings_data['id'].numpy()  # Extract IDs
    q1_embeddings = embeddings_data['q1_feats_bert'].numpy()  # Question1 embeddings
    q2_embeddings = embeddings_data['q2_feats_bert'].numpy()  # Question2 embeddings

    debug_print("Extracted IDs and embeddings from .pt file", {
        "Number of IDs": len(ids),
        "Shape of q1_embeddings": q1_embeddings.shape,
        "Shape of q2_embeddings": q2_embeddings.shape
    })
except Exception as e:
    debug_print("Error while extracting data from .pt file", e)
    raise

# Load df_with_features.csv
try:
    df_with_features = pd.read_csv("Data/df_with_features.csv", encoding='latin-1')
    debug_print("Loaded df_with_features.csv successfully", df_with_features.head())
    debug_print("Shape of df_with_features", df_with_features.shape)  # Shape check
except Exception as e:
    debug_print("Failed to load df_with_features.csv", e)
    raise

# Remove redundant columns from df_with_features
try:
    df_features = df_with_features.drop(['qid1', 'qid2', 'question1', 'question2'], axis=1)
    debug_print("Dropped unnecessary columns from df_with_features", df_features.head())
    debug_print("Shape of df_features after dropping columns", df_features.shape)  # Shape check
except KeyError as e:
    debug_print("KeyError while dropping columns from df_with_features", e)
    raise

# Convert q1_feats_bert and q2_feats_bert into separate DataFrames
try:
    df_embeddings_q1 = pd.DataFrame(q1_embeddings, index=ids, columns=[f'q1_feat_{i}' for i in range(q1_embeddings.shape[1])])
    df_embeddings_q2 = pd.DataFrame(q2_embeddings, index=ids, columns=[f'q2_feat_{i}' for i in range(q2_embeddings.shape[1])])
    debug_print("Converted q1_feats_bert and q2_feats_bert into separate DataFrames", {
        "Head of q1_embeddings DataFrame": df_embeddings_q1.head(),
        "Head of q2_embeddings DataFrame": df_embeddings_q2.head()
    })
    debug_print("Shape of df_embeddings_q1", df_embeddings_q1.shape)  # Shape check
    debug_print("Shape of df_embeddings_q2", df_embeddings_q2.shape)  # Shape check
except Exception as e:
    debug_print("Error while converting embeddings into DataFrames", e)
    raise

# Merge all features
try:
    result = df_features.set_index('id').join(df_embeddings_q1, how='left').join(df_embeddings_q2, how='left')
    debug_print("Merged all features into a single DataFrame", result.head())
    debug_print("Shape of the final merged DataFrame", result.shape)  # Shape check
except Exception as e:
    debug_print("Error while merging features", e)
    raise
# Reset the index to include 'id' as a column
result = df_features.set_index('id').join(df_embeddings_q1, how='left').join(df_embeddings_q2, how='left').reset_index()


print("Is 'id' in the final DataFrame index?", 'id' in result.index.names)
print(result.shape)

# Save the final DataFrame to a CSV file
final_features_path = 'Data/final_features.csv'
try:
    result.to_csv(final_features_path, index=True)
    debug_print(f"Final features saved to {final_features_path}")
except Exception as e:
    debug_print("Error during saving final_features.csv", e)
    raise

print(result.shape)

df = pd.read_csv("Data/final_features.csv", encoding='latin-1')

df = df.applymap(lambda x: np.nan if not pd.api.types.is_numeric_dtype(type(x)) and not isinstance(x, (int, float, np.number)) else x)
print(df.shape)
# Check if there are any NA values in the DataFrame
if df.isna().any().any():
    print("Non-numeric values replaced with NaN.")
else:
    print("No non-numeric values found.")


df = df.drop(columns=['Unnamed: 0'])

# Get the target variable
y_true = df['is_duplicate']
df.drop(['id', 'is_duplicate'], axis=1, inplace=True)
print(df.shape)

# Convert all the features into numeric
cols = list(df.columns)
for i in cols:
    df[i] = pd.to_numeric(df[i], errors='coerce')

# Convert y_true to a list of integers
y_true = list(map(int, y_true.values))

# Display the first few rows of the data
print(df.head())

X_train,X_test, y_train, y_test = train_test_split(df, y_true, stratify=y_true, test_size=0.3)

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

## Random model as baseline

# Set seed for reproducibility
np.random.seed(42)

# Generate random probability predictions for y_train
random_predictions_train = np.random.rand(len(y_train))

# Calculate log loss for y_train
log_loss_train = log_loss(y_train, random_predictions_train)

print(f'Log Loss for Training Data: {log_loss_train:.5f}')

# Generate random probability predictions for y_test
random_predictions_test = np.random.rand(len(y_test))

# Calculate log loss for y_test
log_loss_test = log_loss(y_test, random_predictions_test)

print(f'Log Loss for Test Data: {log_loss_test:.5f}')
print("done")
##
# Call the logistic regression function
logistic_regression_function(X_train, X_test, y_train, y_test)
