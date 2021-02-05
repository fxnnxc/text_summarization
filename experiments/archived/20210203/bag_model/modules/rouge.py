import torch.nn as nn
import torch 
from fairseq.modules import MultiheadAttention
from torch.nn.parameter import Parameter
import torch.nn.functional as F 

class RougePredictNetwork(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(RougePredictNetwork, self).__init__()
        self.num_layers = 2
        self.hidden_size = z_dim
        self.rouge_predict_gru = nn.GRU(x_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.rouge_predict_linear = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        """
        X : Tensor (Tokens, Batch, Dim) 
        """
        x = x.view(x.size(1), x.size(0), x.size(2))
        if False:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(x.device).half()
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        x, hn = self.rouge_predict_gru(x, h0)
        x = self.rouge_predict_linear(x[:, -1, :])
        return x                                # Dim : [B] 
