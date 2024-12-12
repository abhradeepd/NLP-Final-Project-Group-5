import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from data_loader import load_data, TextPairDataset
from model import BertForTextPairClassification
from evaluate import eval_model
from sklearn.model_selection import train_test_split
import numpy as np
def main():

    # Parameters
    DATA_FILE = '../Data/val_QQP_test.csv'  # Replace with your test data file path
    #DATA_FILE = '../Data/final_test.csv'
    PRETRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    MODEL_PATH = 'best_model_state_bert_pair_mlp2_fin.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess test data
    print("Loading test data...")

    df = load_data(DATA_FILE)
    #train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df = df
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)

    test_dataset = TextPairDataset(test_df, tokenizer, max_length=MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize and load the trained model
    print("Loading trained model...")
    model = BertForTextPairClassification(pretrained_model_name=PRETRAINED_MODEL_NAME, drop_out=0.1, num_labels=2, freeze_layers=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)

    # Evaluate the model on the test set
    print("Evaluating the test set...")
    acc, f1_macro, f1_micro, cohen_kappa, avg_loss, auc_score, conf_matrix = eval_model(model, test_loader, device)

    # Print evaluation metrics
    print("\nTest Set Evaluation Results:")
    print(f"Val loss:{avg_loss}")
    print(f"Accuracy: {acc}")
    print(f"F1 Macro Score: {f1_macro}")
    print(f"F1 Micro Score: {f1_micro}")
    print(f"Cohen Kappa: {cohen_kappa}")
    print(f"AUC Score: {auc_score}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save metrics to a file
    with open("evaluation_results.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"F1 Macro Score: {f1_macro}\n")
        f.write(f"F1 Micro Score: {f1_micro}\n")
        f.write(f"Cohen Kappa: {cohen_kappa}\n")
        f.write(f"AUC Score: {auc_score}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix, separator=', '))


if __name__ == '__main__':
    main()
