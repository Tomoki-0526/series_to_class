import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        position_encode = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        position_encode[:, 0::2] = torch.sin(position * div_term)
        position_encode[:, 1::2] = torch.cos(position * div_term)
        position_encode = position_encode.unsqueeze(0).transpose(0, 1)
        self.register_buffer('position_encode', position_encode)

    def forward(self, x):
        x = x + self.position_encode[:x.size(0), :]
        return self.dropout(x)
    

class TransformerClassifier(nn.Module):
    def __init__(self, nhead, nhid, nenclayers, ndeclayers, ninput=4, ntoken=2, dropout=0.5):
        super(TransformerClassifier, self).__init__()

        self.transformer = nn.Transformer(d_model=ninput, nhead=nhead, num_encoder_layers=nenclayers, num_decoder_layers=ndeclayers, dim_feedforward=nhid, dropout=dropout)
        self.positional_encoding = PositionEncoding(d_model=ninput, dropout=0)
        self.predictor = nn.Linear(ninput, ntoken)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.ones(sz, sz)
        mask = torch.triu(mask)
        mask = mask == 1
        mask = mask.transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = src.transpose(0, 1)
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.size()[0])
        
        src = self.positional_encoding(src)
        output = self.transformer(src=src, tgt=src, src_mask=src_mask, tgt_mask=src_mask)
        output = self.predictor(output)
        output = F.log_softmax(output, dim=1)
        return output[-1]
