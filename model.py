import torch
from torch import nn
from torch.nn import functional as F

class PriceNet(nn.Module):
    def __init__(self, input_dim, h_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h_dim//4)
        self.dropout1 = nn.Dropout(p=0.5)
        #self.fc2 = nn.Linear(h_dim//4, h_dim//16)
        #self.dropout0 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(h_dim//4, output_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                m.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        h1 = self.dropout1(F.elu(self.fc1(x)))
        #h2 = F.elu(self.fc2(h1))
        output = self.fc3(h1)

        return output