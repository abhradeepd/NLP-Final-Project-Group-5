import torch
import torch.nn as nn
class BaseModel(nn.Module):
    def __init__(self, head_type="lstm", feature_dim=768, features_dim=40, hidden_dim=128):
        super(BaseModel, self).__init__()
        # Shared fully connected layer for q1 and q2
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU()
        )
        # Fully connected layer for features
        self.features_fc = nn.Linear(features_dim, hidden_dim)
        # Head selection
        if head_type == "lstm":
            self.head = nn.LSTM(3 * hidden_dim, hidden_dim, batch_first=True)
        elif head_type == "gru":
            self.head = nn.GRU(3 * hidden_dim, hidden_dim, batch_first=True)
        elif head_type == "cnn":
            self.head = nn.Sequential(
                nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
        else:
            raise ValueError("Invalid head type. Choose from ['lstm', 'gru', 'cnn']")

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, q1, q2, features):
        # Process shared layers
        q1 = self.shared_fc(q1)  # [batch_size, hidden_dim]
        q2 = self.shared_fc(q2)  # [batch_size, hidden_dim]
        features = self.features_fc(features)  # [batch_size, hidden_dim]

        # Combine features
        combined = torch.cat([q1, q2, features], dim=1)  # [batch_size, 3 * hidden_dim]

        # Pass through the selected head
        if isinstance(self.head, nn.LSTM):
            combined = combined.unsqueeze(1)  # Add sequence dimension
            _, (head_out, _) = self.head(combined)  # LSTM returns (output, (hidden, cell))
            head_out = head_out[-1]  # Take the last hidden state
        elif isinstance(self.head, nn.GRU):
            combined = combined.unsqueeze(1)  # Add sequence dimension
            _, head_out = self.head(combined)  # GRU returns (output, hidden)
            head_out = head_out[-1]  # Take the last hidden state
        elif isinstance(self.head, nn.Sequential):  # CNN
            combined = combined.unsqueeze(2)  # Add a length dimension
            head_out = self.head(combined).squeeze(2)  # Remove the length dimension
        return self.fc_out(head_out)
        # Output
        #return torch.sigmoid(self.fc_out(head_out))

