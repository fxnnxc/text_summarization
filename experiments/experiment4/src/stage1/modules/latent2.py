import torch.nn as nn
import torch 
from fairseq.modules import MultiheadAttention
from torch.nn.parameter import Parameter

class LatentGeneratorNetwork2(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(LatentGeneratorNetwork2, self).__init__()
        self.z_dim = z_dim
        self.mlp_mu_x = nn.Sequential(
                    nn.Linear(x_dim, z_dim))
                   # nn.ReLU(),
                   # nn.Linear(z_dim, z_dim))
        self.mlp_lv_x = nn.Sequential(
                    nn.Linear(x_dim, z_dim))
                  #  nn.ReLU(),
                    # nn.Linear(z_dim, z_dim))
        

    def forward(self, x):
        """
        X : Tensor (Tokens, Batch, Dim) 
        return : Z, Mu, LogVar (BSZ, z_dim, 1) 
        """
        x = x.view(x.size(1), x.size(0), x.size(2))
        xt = x.mean(dim=1)
        xt = xt.unsqueeze(dim=1)
        bsz =  x.size(0)
        x_mu = self.mlp_mu_x(xt)                              # b x z x 1
        x_lv = self.mlp_lv_x(xt)
        x_mu = x_mu.view(bsz, 1, -1)
        x_lv = x_lv.view(bsz, 1, -1)

        x_z = self.reparameterize(x_mu, x_lv)
        x_z = x_z.view(bsz, 1 , -1) 
        return x_z, x_mu, x_lv           # Batch x z_dim x 1
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return  eps * std + mu