import torch
import math
import torch.nn as nn
from torch.distributions import normal
from fairseq.modules.transformer_layer import TransformerEncoderLayer
from fairseq.modules.multihead_attention import MultiheadAttention
import random
from torch.autograd import Variable
from fairseq.modules import TransformerDecoderLayer
from fairseq.models.fairseq_encoder import EncoderOut



def Linear(in_features, out_features, xavier=False, a=-1.0, b=1.0, model=None, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    if xavier:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.uniform_(m.weight, a=a, b=b)
    if bias:
        nn.init.constant_(m.bias, 0.01)
    if model != None:
        m = m.cuda()
    return m

class LSTMBart(nn.Module):
    """lstm bart"""

    def __init__(self, n=1024, basemodel=None, args=None):
        super().__init__()
        self.lstm_layer = nn.LSTM(
            input_size=n,
            hidden_size=n,
            num_layers=1,
            batch_first=True,
            # dropout=0.1,
        )
        # self.final_layer = Linear(int(n), 50264, xavier=True, bias=False)

    def forward(self, features, h, c):
        lstm_features, extra = self.lstm_layer(features, (h,c))
        # out = self.final_layer(lstm_features)
        return lstm_features, extra

class BART_VAE_ANNEALING(nn.Module):
    """Variational Autoencoder"""

    def __init__(self,
                args,
                baseline_model=None, 
                lb_decay_rate=1,
                lb_decay_step=1000,
                tradeoffXZ=0.5,
                n=1024):
        super().__init__()
        self.baseline_model = baseline_model
        self.encoder = baseline_model.encoder 
        self.decoder = baseline_model.decoder
        self.args = args
        self.lstm_out = LSTMBart()
        # self.lstm_mu = nn.LSTM(n,n//2, 2)  #input_dim x hidden x layer
        # self.lstm_var = nn.LSTM(n,n//2, 2)
        self.lstm_mu = nn.GRU(n,n//2, 2)  #input_dim x hidden x layer
        self.lstm_var = nn.GRU(n,n//2, 2)
        self.mlp_mu = nn.Sequential(
                    Linear(n, n//2),
                    nn.ReLU(),
                    Linear(n//2, n)
                )
        self.mlp_var = nn.Sequential(
                    Linear(n, n//2),
                    nn.ReLU(),
                    Linear(n//2, n)
                )
        
        self.n = n
        self.tradeoffXZ = tradeoffXZ
        # Z dim 512 -> 1024
        self.z_transform = nn.Sequential(            
            Linear(int(n/2), int(n), xavier=True),
            nn.Dropout(0.1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
   
    def forward(
                self,
                src_tokens,
                src_lengths,
                prev_output_tokens,
                features_only=False,
                classification_head_name=None,
                token_embeddings=None,
                **kwargs,
            ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(   # T X B X C
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            **kwargs,
        )

        x = encoder_out['encoder_out'][0]
        
        x = x.view((x.size(1), x.size(0), x.size(2)))
        
        h0_m = torch.zeros(2, x.size(1), self.n//2, device='cuda') # Layer x Batch x hidden
        # c0_m = torch.zeros(2, x.size(1), self.n//2, device='cuda')
        h0_v = torch.zeros(2, x.size(1), self.n//2, device='cuda') # Layer x Batch x hidden
        # c0_v = torch.zeros(2, x.size(1), self.n//2, device='cuda')


        # h0_m = torch.zeros(2, x.size(1), self.n//2, device='cuda').half() # Layer x Batch x hidden
        # c0_m = torch.zeros(2, x.size(1), self.n//2, device='cuda').half()
        # h0_v = torch.zeros(2, x.size(1), self.n//2, device='cuda').half() # Layer x Batch x hidden
        # c0_v = torch.zeros(2, x.size(1), self.n//2, device='cuda').half()
        
        # mu, (hn, cn) = self.lstm_mu(x, (h0_m, c0_m))
        # var, (hn, cn) = self.lstm_var(x, (h0_v, c0_v))
        # mu, hn = self.lstm_mu(x, h0_m)
        # var, hn = self.lstm_var(x, h0_v)


        z = self.reparameterize(mu[:,-1,:], var[:, -1, : ])        
        z = self.z_transform(z)
        z = z.unsqueeze(dim=1)
        
        alpha = self.tradeoffXZ
        #xz = alpha*x+(1-alpha)*z
        xz = torch.cat([x,z], dim=1)

        #encoder_out.encoder_out = z.unsqueeze(dim=0)

        encoder_out_ = {
            "encoder_out": [xz],  # T x B x C
            "encoder_padding_mask": [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

        # print(encoder_out["encoder_out"])
        # assert False



        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out_, #encoder_out_,
            features_only=False,
            **kwargs,
        )
        # print('-------------')
        # print(x.size())
        # print('-------------')

        # x = self.output_layer(features)
        # extra['features'] = features
        extra['encoder_out'] = encoder_out

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )

        return (x, extra), mu[:,-1,:], var[:, -1, : ]


    def decoder_vae(self, prev_output_tokens, encoder_out, incremental_state=None):
        n = 1024
        # Original BART decoder features and output: net_output = decoder features
        enc_out = encoder_out.encoder_out
        net_output, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            features_only=True,
        )
        output = self.decoder.output_layer(net_output)

        # Context vector
        ctx_enc = enc_out.transpose(0,1).mean(dim=1)
        ctx_dec = net_output.mean(dim=1)
        # ctx_dec = torch.cat((enc_out.transpose(0,1), net_output), dim=1).mean(dim=1)

        # Mean and log variance
        ms_enc = self.musig_enc(ctx_enc)
        ms_dec = self.musig_dec(ctx_dec)#.squeeze(dim=1)
        mean_enc, lvar_enc = torch.split(ms_enc, int(n/2), dim=-1)
        mean_dec, lvar_dec = torch.split(ms_dec, int(n/2), dim=-1)
        dec_shape = mean_dec.shape

        # Generate latent variable z
        z_dec = self.sample(dec_shape, mean_enc, lvar_enc).unsqueeze(dim=1)
        z_dec = self.z_transform(z_dec)

        # Combine z with BART features to extract new VAE features and VAE vocab distribution
        lstm_features, _ = self.lstm_out(net_output, z_dec.transpose(0,1), z_dec.transpose(0,1))
        out_vae = self.decoder.output_layer(lstm_features)

        # Miscellaneous
        extra['kl'] = (mean_enc, lvar_enc, mean_dec, lvar_dec)
        return (out_vae, extra)

    def generate_probs(self, embeds, z, output, self_attn_mask, self_attn_padding_mask):
        gens, _, _ = self.cross_layer(
            embeds.transpose(0,1),
            z.transpose(0,1),
            None,
            None,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=False,
            need_head_weights=False,
        )
        # combined = torch.cat((gens.transpose(0,1), output), dim=-1)
        combined = torch.cat((gens.transpose(0,1), output), dim=1)
        lstm_features, _ = self.lstm_out(combined)
        out_vae = self.decoder.output_layer(lstm_features)
        return out_vae, lstm_features

    def sample(self, shape, mean, lvar):
        norm_dist = normal.Normal(0., 1.)
        eps = norm_dist.sample(shape).cuda().detach()#.half()
        z = torch.exp(0.5 * lvar) * eps + mean
        return z

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)
        return layer

    def teacher_forcing(self, z, src_tokens, net_output, prev_output_tokens, extra):
        lamb = 1
        max_len = prev_output_tokens.size(1)
        # num = random.sample(range(0,max_len-1), 1)[0]
        # idxs = list(range(num, min(num + 10, max_len)))
        output = extra['output']
        gens = self.cross_attention(gens=z, embeds=extra['decoder_embedding'])
        """ -------------------------- """
        # for i, step in enumerate(idxs):
        #     # Teacher forcing
        #     gens = self.cross_attention(gens=gens, embeds=extra['decoder_embedding'][:, :step+1])
        #     gens = self.get_info(gens, extra['encoder_out'].encoder_out, net_output, src_tokens, prev_output_tokens)
            # out = self.mlp(combined)
            # output[:, i, :] = lamb * out + (1-lamb) * output[:, i, :]
        """ -------------------------- """
        combined = torch.cat((gens, net_output), dim=-1)
        out_vae, _ = self.lstm_out(combined)
        output = lamb * out_vae + (1 - lamb) * output
        return output

    def get_info(self, gens, enc_infos, dec_infos, tokens_enc, tokens_dec):
        bsz = dec_infos.size(0)
        length = gens.size(1)
        if gens.size(0) != bsz:
            gens = gens.repeat(bsz, 1, 1)
        encoder_padding_mask = tokens_enc.eq(1)
        gens = self.enc_attn(
            query=gens.transpose(0,1),
            key=enc_infos,
            value=enc_infos,
            key_padding_mask=encoder_padding_mask,
            static_kv=True,
            need_weights=False,
        )[0]

        decoder_padding_mask = tokens_dec[:, :length].eq(1)
        gens = self.dec_attn(
            query=gens,
            key=dec_infos[:, :length, :].transpose(0,1),
            value=dec_infos[:, :length, :].transpose(0,1),
            key_padding_mask=decoder_padding_mask,
            static_kv=True,
            need_weights=False,
        )[0]
        return gens.transpose(0,1)
    
    def cross_attention(self, gens, embeds):
        gens = gens.repeat(embeds.size(0), 1, 1)
        gens = self.cross_attn(
            query=embeds.transpose(0,1),
            key=gens.transpose(0,1),
            value=gens.transpose(0,1),
            key_padding_mask=None,
            static_kv=True,
            need_weights=False,
        )[0].transpose(0,1)
        return gens

    def decoder_embed(self, prev_output_tokens, incremental_state=None):
        # embed positions
        positions = (
            self.decoder.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
        )
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        self.decoder.embed_scale = math.sqrt(self.n)
        x = self.decoder.embed_scale * self.decoder.embed_tokens(prev_output_tokens)

        if self.decoder.quant_noise is not None:
            x = self.decoder.quant_noise(x)

        if self.decoder.project_in_dim is not None:
            x = self.decoder.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.decoder.layernorm_embedding is not None:
            x = self.decoder.layernorm_embedding(x)

        x = self.decoder.dropout_module(x)
        return x
    
    def buffered_future_mask(self, x):
        return self.decoder.buffered_future_mask(x)

    def max_positions(self):
        return self.baseline_model.max_positions()
    
    def set_num_updates(self, update_num):
        return self.baseline_model.set_num_updates(update_num)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        return self.baseline_model.get_normalized_probs(net_output, log_probs, sample)

    def get_targets(self, sample, net_output):
        return self.baseline_model.get_targets(sample, net_output)

    def prepare_for_inference_(self, cfg):
        return self.baseline_model.prepare_for_inference_(cfg)

    def max_decoder_positions(self):
        return self.baseline_model.max_decoder_positions()