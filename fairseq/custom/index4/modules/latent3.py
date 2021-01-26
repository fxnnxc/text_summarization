import torch.nn as nn
import torch 
from fairseq.modules import MultiheadAttention
from torch.nn.parameter import Parameter

class LatentGeneratorNetwork(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(LatentGeneratorNetwork, self).__init__()

        self.lstm = nn.LSTM(x_dim, self.hidden_size, self.num_layers , batch_first=True)
        self.mlp_mu_x = nn.Sequential(
                    nn.Linear(self.hidden_size, z_dim),
                    nn.ReLU(),
                    nn.Linear(z_dim, z_dim))
        self.mlp_lv_x = nn.Sequential(
                    nn.Linear(self.hidden_size, z_dim),
                    nn.ReLU(),
                    nn.Linear(z_dim, z_dim))


    def forward(self, x):
        """
        X : input tensor        (tokens, bsz, xdim) 
        return : Z, Mu, LogVar  (bsz, zdim, 1) 
        """
        bsz =  x.size(1)
        x = x.view(x.size(1), x.size(0), x.size(2))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        x = x[:, :x.size(1)//2, :]
        xt = x.mean(dim=1)
        xt = xt.unsqueeze(dim=1)
        
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