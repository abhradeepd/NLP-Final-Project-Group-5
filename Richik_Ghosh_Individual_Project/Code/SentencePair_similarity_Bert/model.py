import torch.nn as nn
from transformers import BertModel, AutoModel
import torch
class BertForTextPairClassification(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_labels=2, drop_out=0.5, freeze_layers=10):
        super(BertForTextPairClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        # Get the BERT hidden size
        bert_hidden_size = self.bert.config.hidden_size

        # Sizes for the hidden layers
        hidden_size1 = bert_hidden_size // 2
        hidden_size2 = bert_hidden_size // 4

        # Dropout for regularization
        self.drop = nn.Dropout(p=drop_out)

        # First hidden layer
        self.fc1 = nn.Linear(bert_hidden_size, hidden_size1)
        self.drop1 = nn.Dropout(p=drop_out)

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.drop2 = nn.Dropout(p=drop_out)

        # Output layer
        self.out = nn.Linear(hidden_size2, num_labels)

        # Freeze the first `freeze_layers` layers
        for idx, layer in enumerate(self.bert.encoder.layer):
            if idx < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        # Pass inputs through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation

        # Pass through the first hidden layer
        hidden_output1 = torch.relu(self.fc1(self.drop(pooled_output)))
        hidden_output1 = self.drop1(hidden_output1)

        # Pass through the second hidden layer
        hidden_output2 = torch.relu(self.fc2(hidden_output1))
        hidden_output2 = self.drop2(hidden_output2)

        # Output layer
        logits = self.out(hidden_output2)
        #logits = self.out(hidden_output2).squeeze(-1) # shape: [batch_size] for bcelogitloss
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            #loss_fct = nn.BCEWithLogitsLoss() # started overfitting
            loss = loss_fct(logits, labels) #float is done for bcelogitloss

        return loss, logits
# class SBERTForTextPairClassification(nn.Module):
#     def __init__(self, pretrained_model_name='sentence-transformers/all-mpnet-base-v2', num_labels=2, drop_out=0.5):
#         super(SBERTForTextPairClassification, self).__init__()
#         self.sbert = AutoModel.from_pretrained(pretrained_model_name)
#         sbert_hidden_size = self.sbert.config.hidden_size
#
#         # Dropout for regularization
#         self.drop = nn.Dropout(p=drop_out)
#
#         # Classification head
#         self.fc = nn.Linear(sbert_hidden_size * 3, num_labels)
#
#         # Freeze all layers except the last two
#         for idx, layer in enumerate(self.sbert.encoder.layer):
#             if idx < len(self.sbert.encoder.layer) - 2:  # Keep only the last 2 layers trainable
#                 for param in layer.parameters():
#                     param.requires_grad = False
#
#     def forward(self, input_ids, attention_mask, labels=None):
#         # Extract embeddings for sentence pairs
#         outputs = self.sbert(input_ids=input_ids, attention_mask=attention_mask)
#         sentence_embeddings = outputs.pooler_output  # Pooling output for [CLS] token representation
#
#         # Check for even batch size
#         batch_size = sentence_embeddings.size(0)
#         if batch_size % 2 != 0:
#             sentence_embeddings = sentence_embeddings[:batch_size - 1]  # Truncate if odd
#
#         # Compute concatenation of embeddings
#         emb_1, emb_2 = torch.chunk(sentence_embeddings, 2, dim=0)
#         combined_emb = torch.cat([emb_1, emb_2, torch.abs(emb_1 - emb_2)], dim=1)
#
#         # Pass through dropout and classification head
#         logits = self.fc(self.drop(combined_emb))
#
#         # Calculate loss if labels are provided
#         loss = None
#         if labels is not None:
#             labels = labels[:logits.size(0)]  # Adjust labels
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits, labels)
#
#         return loss, logits




