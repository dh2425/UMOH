import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
class GraphConvolution(nn.Module):
    '''
        GCN layer.
    '''
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        output = torch.mm(input, self.weight)
        output = torch.mm(adj, output)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNet_IMG(nn.Module):
    def __init__(self, bit, gamma, batch_size):
        super(GCNet_IMG, self).__init__()

        self.gc1 = GraphConvolution(512, 2048)
        self.gc2 = GraphConvolution(2048, 1024)
        self.linear = nn.Linear(1024, bit)
        self.alpha = 1.0
        self.gamma = gamma
        self.weight = nn.Parameter(torch.FloatTensor(batch_size, batch_size))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x, adj):
        #原始图结构的某种“修正”或“增强”。
        adj = adj + self.gamma * self.weight  #gamma 0.45  weight(32 32)
        x = torch.relu(self.gc1(x, adj))
        x = torch.relu(self.gc2(x, adj))
        x = self.linear(x)
        code = torch.tanh(self.alpha * x)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class PositionalEncoding(nn.Module):
    """
    Sin-cos position embedding
    LND - LND
    """

    def __init__(self, d_model, dropout=0., max_len=128):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # [max-length, 1, d_model]
        pe = pe / (d_model ** 0.5)  #
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: LND
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

