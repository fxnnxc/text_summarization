import torch.nn as nn
import torch 
from fairseq.modules import MultiheadAttention
from torch.nn.parameter import Parameter

class LatentGeneratorNetwork(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(LatentGeneratorNetwork, self).__init__()
        self.z_dim = z_dim
        self.x_to_mu = Parameter(torch.randn((z_dim, 1, x_dim), dtype=torch.float32, device='cuda', requires_grad=True))
        self.x_to_lv = Parameter(torch.randn((z_dim, 1, x_dim), dtype=torch.float32, device='cuda', requires_grad=True))
        self.mult_attn_x_mu = MultiheadAttention(x_dim, 1, 
                                    kdim=x_dim, vdim=x_dim, encoder_decoder_attention=True)
        self.mult_attn_x_lv = MultiheadAttention(x_dim, 1, 
                                    kdim=x_dim, vdim=x_dim, encoder_decoder_attention=True)
        self.mlp_mu_x = nn.Sequential(
                    nn.Linear(x_dim, z_dim),
                    nn.ReLU(),
                    nn.Linear(z_dim, 1))
        self.mlp_lv_x = nn.Sequential(
                    nn.Linear(x_dim, z_dim),
                    nn.ReLU(),
                    nn.Linear(z_dim, 1))
        #self.mlp_xz = nn.Sequential(nn.Linear(z_dim, x_dim))
        

    def forward(self, x):
        """
        X : Tensor (Tokens, Batch, Dim) 
        return : Z, Mu, LogVar (BSZ, z_dim, 1) 
        """
        # x = x.view(x.size(2), -1, x.size(0))
        z_dim, bsz, x_dim = self.z_dim, x.size(1), x.size(2)
        x_to_mu_exp_bsz = self.x_to_mu.expand(z_dim, bsz, x_dim)
        x_to_lv_exp_bsz = self.x_to_lv.expand(z_dim, bsz, x_dim)
        x_mu = self.mult_attn_x_mu(x_to_mu_exp_bsz, x, x)[0]    # z x b x xdim 
        x_lv = self.mult_attn_x_lv(x_to_lv_exp_bsz, x, x)[0]    # z x b x xdim

        x_mu = x_mu.view(bsz, z_dim, x_dim)                     # b x z x xdim
        x_lv = x_lv.view(bsz, z_dim, x_dim)
        x_mu = self.mlp_mu_x(x_mu)                              # b x z x 1
        x_lv = self.mlp_lv_x(x_lv)
        x_mu = x_mu.view(bsz, 1, -1)
        x_lv = x_lv.view(bsz, 1, -1)

        x_z = self.reparameterize(x_mu, x_lv)
        x_z = x_z.view(bsz, 1 , -1) 
        
        return x_z, x_mu, x_lv           # Batch x z_dim x 1
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return  eps * std + mu