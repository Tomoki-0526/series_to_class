import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=6, onehot_size=2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, onehot_size)

    def forward(self, sequence):
        x = sequence # (batch_size, seq_length, vector_length)
        x = x.transpose(0, 1) # (seq_l, batch_size, vector_l)
        out, _ = self.lstm(x)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        # print(out.shape)
        return out[-1]
