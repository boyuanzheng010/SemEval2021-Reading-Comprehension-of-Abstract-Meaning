import torch.nn as nn
import torch.nn.functional as F
import torch

from Baselines.Models.Linear import Linear

class MLPAttention(nn.Module):

    def __init__(self, dim, dropout):
        super(MLPAttention, self).__init__()

        self.Q_W = Linear(dim, dim)
        self.K_W = Linear(dim, dim)
        self.V_W = Linear(dim, dim)

        self.tanh = nn.Tanh()
        self.V = Linear(dim, 1)

        self.dropout = nn.Dropout(dropout)


    def forward(self, Q, K, V):
        # Q: [batch_size, dim]
        # K: [batch_size, seq_len, dim]
        # V: [batch_size, seq_len, dim]

        Q = self.dropout(self.Q_W(Q)) # [batch_size, dim]
        K = self.dropout(self.K_W(K)) # [batch_size, seq_len, dim]
        V = self.dropout(self.V_W(V)) # [batch_size, seq_len, dim]

        Q = Q.unsqueeze(1) # [batch_size, 1, dim]
        M = self.dropout(self.tanh(Q + K))  # [batch_size, seq_len, dim]
        scores = self.dropout(self.V(M)) # [batch_size, seq_len, 1]
        scores = F.softmax(scores, dim=1) # [batch_size, seq_len, 1]

        R = self.dropout(V * scores) # [batch_size, seq_len, dim]

        feat = torch.sum(R, dim=1)  # [batch_size, dim]

        return feat


        






