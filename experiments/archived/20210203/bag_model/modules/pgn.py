import torch.nn as nn
import torch 
from fairseq.modules import MultiheadAttention
from torch.nn.parameter import Parameter
import torch.nn.functional as F 

class ProbGeneratorNetwork(nn.Module):
    def __init__(self, x_dim, v_dim):
        super(ProbGeneratorNetwork, self).__init__()
        self.XtoV = nn.Sequential(
                    nn.Linear(x_dim, v_dim))

    def forward(self, x):
        """
        X : Tensor (Tokens, Batch, Dim) 
        return : Z, Mu, LogVar (BSZ, z_dim, 1) 
        """
        x = x.view(x.size(1), x.size(0), x.size(2))
        x = self.XtoV(x)                              # b x z x v
        x = F.log_softmax(x, dim=-1)
        x = x.mean(dim=1)
        x = x.unsqueeze(dim=1)
        return x                                # Batch x z_dim x 1
