# Simple VAE

## Model 

```python
def __init__(self, args, pretrained_bart):
        super().__init__(args, pretrained_bart.encoder, pretrained_bart.decoder)
        self.encoder = pretrained_bart.encoder 
        self.decoder = pretrained_bart.decoder
        self.mlp_mu = nn.Sequential(
                    nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim//2),
                    nn.ReLU(),
                    nn.Linear(args.encoder_embed_dim//2, args.encoder_embed_dim)
                )
        self.mlp_var = nn.Sequential(
                    nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim//2),
                    nn.ReLU(),
                    nn.Linear(args.encoder_embed_dim//2, args.encoder_embed_dim)
                )
        self.mult_attn = MultiheadAttention(args.encoder_embed_dim, 2, 
                                kdim=args.encoder_embed_dim, vdim=args.encoder_embed_dim, 
                                encoder_decoder_attention=True)
```

## Forward 
```python
#1 encoder
encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens
        )

x = encoder_out['encoder_out'][0]
xt = x.view((x.size(1), x.size(0), x.size(2))) # change shape
#2 take mean
xt = xt.mean(dim=1)

#3 generate z
mu = self.mlp_mu(xt)
var= self.mlp_var(xt)
z = self.reparameterize(mu, var)     

#4 Cross Attention
xz = self.mult_attn(x, z, z)[0]
encoder_out['encoder_out'][0] = xz

#5 Decoder
x, extra = self.decoder(encoder_out)

return x
```



## Loss 

```python
 x, extra = net_output
mu , log_var = extra['mu'], extra['var']
kld_loss = torch.mean(-0.5 * torch.sum(1+ log_var - mu**2- log_var.exp(), dim=1), dim=-1)
elbo = loss + kld_loss*sample_size
```


## Train Schedule

```python
(loss,kld_loss), sample_size, logging_output = criterion(model, sample)
            FREQ = 200
            R = 0.99
            tau = (update_num%FREQ)/FREQ
            beta = (tau/R)**5/10 if tau<=R else 1/2    
            elbo = loss + beta*kld_loss
```