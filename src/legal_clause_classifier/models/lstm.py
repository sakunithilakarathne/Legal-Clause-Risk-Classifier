import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, pad_idx=0, num_layers=1, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_labels)
        self.dropout = nn.Dropout(0.3)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)  # [B, T, E]
        packed_out, (h_n, _) = self.lstm(embedded)

        if self.lstm.bidirectional:
            pooled = torch.cat((h_n[-2], h_n[-1]), dim=1)  # concat forward/backward
        else:
            pooled = h_n[-1]

        x = self.dropout(pooled)
        logits = self.fc(x)
        return logits