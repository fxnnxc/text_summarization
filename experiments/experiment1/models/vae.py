import torch
import math
import torch.nn as nn
from torch.distributions import normal
from fairseq.modules.transformer_layer import TransformerEncoderLayer
from fairseq.modules.multihead_attention import MultiheadAttention
import random
from torch.autograd import Variable
from fairseq.modules import TransformerDecoderLayer

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
            hidden_size=int(n/2),
            num_layers=1,
            batch_first=True,
            # dropout=0.1,
        )
        # self.final_layer = Linear(int(n), 50264, xavier=True, bias=False)

    def forward(self, features, h, c):
        lstm_features, extra = self.lstm_layer(features, (h,c))
        # out = self.final_layer(lstm_features)
        return lstm_features, extra

class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(self,
                args,
                baseline_model=None, 
                lb_decay_rate=1,
                lb_decay_step=1000,
                n=768):
        super().__init__()
        self.baseline_model = baseline_model
        self.args = args
        # self.lstm_out = LSTMBart()
        self.n = n
        self.z_dim = int(n/4)

        # Mean and variance (Log variance in this case)
        self.musig_enc = nn.Sequential(
            Linear(int(n), self.z_dim, xavier=True),
            nn.GELU(),
            Linear(self.z_dim, self.z_dim*2, xavier=True),
            nn.Dropout(0.1)
        )
        self.musig_dec = nn.Sequential(
            Linear(int(n), self.z_dim, xavier=True),
            nn.GELU(),
            Linear(self.z_dim, self.z_dim*2, xavier=True),
            nn.Dropout(0.1)
        )
        self.proj = Linear(self.z_dim + n, 51200, xavier=True)

        # Z dim 512 -> 1024
        # self.z_transform = nn.Sequential(            
        #     Linear(int(n/2), int(n), xavier=True),
        #     nn.Dropout(0.1)
        # )
        # self.proj = Linear(int(n/2), 50264, xavier=True)
        
    def forward(self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        features_only=False,
        classification_head_name=None,
        token_embeddings=None,
        generation=False,
        teacher=False,
        **kwargs,
    ):
        # Extract encoder output -> (encoder_out, ...): # D x B x dim
        encoder_out = self.baseline_model.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )

        # Extract vocab distribution by VAE # B x S x Vocab
        (out_vae, extra) = self.decoder_vae(prev_output_tokens, encoder_out)
        return (out_vae, extra)

    def decoder_vae(self, prev_output_tokens, encoder_out, incremental_state=None):
        # n = 512
        # Original BART decoder features and output: net_output = decoder features
        enc_out = encoder_out.encoder_out
        net_output, extra = self.baseline_model.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            features_only=True,
        )
        output = self.baseline_model.decoder.output_layer(net_output)

        # Context vector
        ctx_enc = enc_out.transpose(0,1).mean(dim=1)
        # ctx_dec = net_output.mean(dim=1)
        # ctx_dec = torch.cat((enc_out.transpose(0,1), net_output), dim=1).mean(dim=1)

        # Mean and log variance
        ms_enc = self.musig_enc(ctx_enc)
        ms_dec = self.musig_dec(ctx_dec)#.squeeze(dim=1)
        mean_enc, lvar_enc = torch.split(ms_enc, self.z_dim, dim=-1)
        mean_dec, lvar_dec = torch.split(ms_dec, self.z_dim, dim=-1)
        dec_shape = mean_dec.shape

        # Generate latent variable z
        z_dec = self.sample(dec_shape, mean_dec, lvar_dec).unsqueeze(dim=1)
        # z_dec = self.z_transform(z_dec)

        # Combine z with BART features to extract new VAE features and VAE vocab distribution
        zs = z_dec.repeat(1, net_output.size(1), 1)
        final_feat = torch.cat((net_output, zs), dim=-1)
        out_vae = self.proj(final_feat)

        ''' LSTM '''
        # attn = extra['attn'][0]
        # enc_attn = torch.matmul(attn, enc_out.transpose(0,1))
        # lstm_features, _ = self.lstm_out(enc_attn, z_dec.transpose(0,1), z_dec.transpose(0,1))
        # out_vae = self.proj(lstm_features)
        # out_vae = self.baseline_model.decoder.output_layer(lstm_features)

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
        out_vae = self.baseline_model.decoder.output_layer(lstm_features)
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
            self.baseline_model.decoder.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
        )
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        self.baseline_model.decoder.embed_scale = math.sqrt(self.n)
        x = self.baseline_model.decoder.embed_scale * self.baseline_model.decoder.embed_tokens(prev_output_tokens)

        if self.baseline_model.decoder.quant_noise is not None:
            x = decoder.quant_noise(x)

        if self.baseline_model.decoder.project_in_dim is not None:
            x = self.baseline_model.decoder.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.baseline_model.decoder.layernorm_embedding is not None:
            x = self.baseline_model.decoder.layernorm_embedding(x)

        x = self.baseline_model.decoder.dropout_module(x)
        return x
    
    def buffered_future_mask(self, x):
        return self.baseline_model.decoder.buffered_future_mask(x)

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