from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import torch
import torch.nn as nn
from contrastive_loss_bert import SiameseQuestionsDataset, SiameseBertModel
import pandas as pd
def classification_results(model, dataloader, device, threshold=0.5):
    """
    Evaluate the trained Siamese model for classification results.
    Generates confusion matrix and classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Classification Results"):
            q1_input_ids = batch['q1_input_ids'].to(device)
            q1_attention_mask = batch['q1_attention_mask'].to(device)
            q2_input_ids = batch['q2_input_ids'].to(device)
            q2_attention_mask = batch['q2_attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Get embeddings from Siamese model
            q1_embeddings, q2_embeddings = model(q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask)
            cos_sim = nn.CosineSimilarity(dim=1)(q1_embeddings, q2_embeddings).cpu().numpy()

            # Apply threshold to convert similarity scores into binary predictions
            preds = (cos_sim >= threshold).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Not Duplicate", "Duplicate"]))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["Not Duplicate", "Duplicate"],
                yticklabels=["Not Duplicate", "Duplicate"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# ============================================================
# Main Function to Evaluate Classification Results
# ============================================================
def evaluate_classification():
    # Load test data
    test_data_path = "../Data/test_data.csv"  # Ensure this path matches your saved test data
    test_df = pd.read_csv(test_data_path)
    test_dataset = SiameseQuestionsDataset(
        questions1=test_df['question1'].values,
        questions2=test_df['question2'].values,
        labels=test_df['is_duplicate'].values,
        tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
        max_len=128
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load trained model
    model_path = "siamese_bert_model_best_contrastiveLoss_3lay.pth"
    model = SiameseBertModel(bert_model_name='bert-base-uncased', fine_tune_layers=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])

    # Evaluate classification results
    classification_results(model, test_loader, device)

if __name__ == "__main__":
    evaluate_classification()
