import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLSTM(nn.Module):
    def __init__(self, ninput=4, nhid=6, nlayers=1):
        super(MyLSTM, self).__init__()
        
        self.drop = nn.Dropout(0.5)
        self.W = nn.Parameter(torch.zeros(ninput, nhid * 4))
        self.U = nn.Parameter(torch.zeros(nhid, nhid * 4))
        self.b = nn.Parameter(torch.zeros(nhid * 4))
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, state=None):
        input = self.drop(input)
        seq_size, batch_size, _ = input.size()
        if state is None:
            h_t = torch.zeros(batch_size, self.nhid)
            c_t = torch.zeros(batch_size, self.nhid)
        else:
            h_t, c_t = state

        output = []
        nhid = self.nhid
        for t in range(seq_size):
            x = input[t, :, :]
            gates = x @ self.W + h_t @ self.U + self.b
            i, f, g, o = (
                torch.sigmoid(gates[:, :nhid]),
                torch.sigmoid(gates[:, nhid:nhid*2]),
                torch.tanh(gates[:, nhid*2:nhid*3]),
                torch.sigmoid(gates[:, nhid*3:]),
            )
            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            output.append(h_t.unsqueeze(0))
        
        output = torch.cat(output, dim=0).contiguous()
        output = self.drop(output)
        return output, (h_t, c_t)
    
class MyLSTMClassifier(nn.Module):
    def __init__(self, ninput=4, nhid=6, nclasses=2):
        super(MyLSTMClassifier, self).__init__()
        self.hidden_dim = nhid
        self.lstm = MyLSTM(ninput, nhid)
        self.fc = nn.Linear(nhid, nclasses)

    def forward(self, input):
        x = input # (B,L,E)
        x = x.transpose(0, 1) # (L,B,E)
        out, _ = self.lstm(x)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out[-1]