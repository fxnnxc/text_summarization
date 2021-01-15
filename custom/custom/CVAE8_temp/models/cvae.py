from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    base_architecture,
)
from fairseq.models.bart.model import BARTModel
from fairseq import checkpoint_utils
from .hub_interface import VAEBARTHubInterface
from fairseq.models.bart.hub_interface import BARTHubInterface
import os 
import torch 
import torch.nn as nn

from typing import Optional
from fairseq.modules import MultiheadAttention
from torch.autograd import Variable

@register_model("CVAE")
class CVaeBART(BARTModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        BARTModel.add_args(parser)
        parser.add_argument('--pretrained-checkpoint',
                            help='path to load checkpoint from pretrained model')
        parser.add_argument('--pretrained', action="store_true",
                            help='use pretrained model when training [True, ...]')

    def __init__(self, args, pretrained_bart):
        super().__init__(args, pretrained_bart.encoder, pretrained_bart.decoder)
        self.encoder = pretrained_bart.encoder 
        self.decoder = pretrained_bart.decoder
        C = 16# Compression
        x_dim = args.encoder_embed_dim
        self.z_dim=C
        self.x_dim=x_dim
        self.expand_x_dim = x_dim+1
        ex_dim = self.expand_x_dim

        self.x_to_mu = torch.randn((ex_dim, 1, C), dtype=torch.float32, device='cuda', requires_grad=True)
        self.x_to_lv = torch.randn((ex_dim, 1, C), dtype=torch.float32, device='cuda', requires_grad=True)
        self.y_to_mu = torch.randn((x_dim, 1, C), dtype=torch.float32, device='cuda', requires_grad=True)
        self.y_to_lv = torch.randn((x_dim, 1, C), dtype=torch.float32, device='cuda', requires_grad=True)
        


        self.mult_attn_x_mu = MultiheadAttention(C, 1, 
                kdim=ex_dim, vdim=ex_dim, encoder_decoder_attention=True)
        self.mult_attn_x_lv = MultiheadAttention(C, 1, 
                kdim=ex_dim, vdim=ex_dim, encoder_decoder_attention=True)
        self.mult_attn_y_mu = MultiheadAttention(C, 1, 
                kdim=x_dim, vdim=x_dim, encoder_decoder_attention=True)
        self.mult_attn_y_lv = MultiheadAttention(C, 1, 
                kdim=x_dim, vdim=x_dim, encoder_decoder_attention=True)
        
        self.mlp_expand_x = nn.Sequential(
                    nn.Linear(x_dim, ex_dim)
                    )
        self.mlp_mu_x = nn.Sequential(
                    nn.Linear(ex_dim, C),
                    nn.ReLU(),
                    nn.Linear(C, C)
                )
        self.mlp_lv_x = nn.Sequential(
                    nn.Linear(ex_dim, C),
                    nn.ReLU(),
                    nn.Linear(C, C)
                )
        self.mlp_mu_y = nn.Sequential(
                nn.Linear(x_dim, C),
                nn.ReLU(),
                nn.Linear(C, C)
            )
        self.mlp_lv_y = nn.Sequential(
                nn.Linear(x_dim, C),
                nn.ReLU(),
                nn.Linear(C, C)
            )
        self.DtoV = nn.Linear(x_dim+C**2, len(self.decoder.dictionary))
        nn.init.normal_(self.DtoV.weight, mean=0, std=1024 ** -0.5)
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens
        )

        x = encoder_out['encoder_out'][0]
        
        output_lengths=torch.tensor([prev_output_tokens.size(1) for i in range(src_lengths.size(0))]).cuda()

        encoder_out2 = self.encoder(
            prev_output_tokens,
            src_lengths=output_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens
        )

        y = encoder_out2['encoder_out'][0]
       
        z_dim, bsz, x_dim, ex_dim = self.z_dim, x.size(1), self.x_dim, self.expand_x_dim
        

        x = x.view(bsz, -1, x_dim)
        print(x.size())
        x = self.mlp_expand_x(x)
        x = x.view(-1, bsz, ex_dim)
        print(x.size())

        x_to_mu_exp_bsz = self.x_to_mu.expand(ex_dim, bsz, z_dim)
        x_to_lv_exp_bsz = self.x_to_lv.expand(ex_dim, bsz, z_dim)
        y_to_mu_exp_bsz = self.y_to_mu.expand(x_dim, bsz, z_dim)
        y_to_lv_exp_bsz = self.y_to_lv.expand(x_dim, bsz, z_dim)

        x_mu = self.mult_attn_x_mu(x_to_mu_exp_bsz, x, x)[0]    # z x b x xdim 
        x_lv = self.mult_attn_x_lv(x_to_lv_exp_bsz, x, x)[0]    # z x b x xdim 
        y_mu = self.mult_attn_y_mu(y_to_mu_exp_bsz, y, y)[0]    # z x b x xdim 
        y_lv = self.mult_attn_y_lv(y_to_lv_exp_bsz, y, y)[0]    # z x b x xdim 
    
        x_mu = x_mu.view(bsz, z_dim, ex_dim)   # b x z x x_dim
        x_lv = x_lv.view(bsz, z_dim, ex_dim)
        y_mu = y_mu.view(bsz, z_dim, x_dim)
        y_lv = y_lv.view(bsz, z_dim, x_dim) 

        x_mu = self.mlp_mu_x(x_mu)        # b x z x 1
        x_lv = self.mlp_lv_x(x_lv)
        y_mu = self.mlp_mu_y(y_mu)
        y_lv = self.mlp_lv_y(y_lv)

        x_z = self.reparameterize(x_mu, x_lv)
        y_z = self.reparameterize(y_mu, y_lv)
        
        x, extra = self.decoder(               # B x T x D
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=True, #features_only, #True,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        
        # concat z to x and generate last prob
        x_z1 = x_z.view(bsz, 1, -1)
        # x_z2 = x_z1.expand(bsz, x.size(1), x_z1.size(2))        # B x 1 x Z**2
        x_z2 = x_z1.repeat(1, x.size(1), 1)
        x = torch.cat([x, x_z2], dim=-1)  # B x T x (x_dim +Z**2) 
        x = self.DtoV(x)   # B x V
        # x = self.softmax(x)
        
        eos: int = self.eos
        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(eos), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break

        extra['x_z']  = x_z
        extra['x_mu'] = x_mu
        extra['x_lv'] = x_lv

        extra['y_z']  = y_z
        extra['y_mu'] = y_mu
        extra['y_lv'] = y_lv
    
        return x, extra                                     #), (mu[:,-1,:], var[:, -1, : ])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        #eps = torch.randn_like(std)
        return mu + std #eps * std + mu

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        sample_break_mode="eos",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            sample_break_mode=sample_break_mode,
            **kwargs,
        )
        return VAEBARTHubInterface(x["args"], x["task"], x["models"][0])


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        trained_model= None
        pretrained = args.pretrained
        if pretrained:
            trained_model = checkpoint_utils.load_model_ensemble(
                filenames=[args.pretrained_checkpoint],
                task=task,
            )[0][0]


            # trained_decoder = list(trained_model.children())[1]
            # trained_encoder = list(trained_model.children())[0]
            # freeze pretrained model
            # for param in trained_decoder.parameters():
            #     param.requires_grad = False
            # for param in trained_encoder.parameters():
            #     param.requires_grad = False
        model = CVaeBART(args, trained_model)

        return model 
        


@register_model_architecture("CVAE", "CVAE_large")
def CVAE_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

@register_model_architecture("CVAE","CVAE_base" )
def CVAE_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    CVAE_large_architecture(args)
