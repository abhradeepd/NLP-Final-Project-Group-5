import pandas as pd
import numpy as np
import torch

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