from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    base_architecture,
)
from fairseq.models.bart.model import BARTModel
from fairseq import checkpoint_utils
from .hub_interface import VAEBARTHubInterface
import os 
import torch 
import torch.nn as nn

from typing import Optional
from fairseq.modules import MultiheadAttention
from torch.autograd import Variable

from ..modules.pgn import ProbGeneratorNetwork
from ..modules.rouge import RougePredictNetwork

import copy

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_model("BAG")
class CVaeBART(BARTModel):
    freeze_encoder = False
    freeze_decoder = False
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        BARTModel.add_args(parser)
        parser.add_argument('--pretrained-checkpoint',
                            help='path to load checkpoint from pretrained model')
        parser.add_argument('--pretrained', action="store_true",
                            help='use pretrained model when training [True, ...]')
        parser.add_argument('--latent-dim', 
                            help='use pretrained model when training [True, ...]')
        parser.add_argument('--latent-alpha',
                            help='use pretrained model when training [True, ...]')

    def __init__(self, args, encoder, decoder, pretrained_encoder=None, pretrained_decoder=None):
        super().__init__(args, encoder, decoder)
        
        self.decvo = True 
        self.new_linear = False
        self.sep_softmax = False

        if pretrained_encoder:
            self.encoder = pretrained_encoder
            self.decoder = pretrained_decoder
        else:
            self.encoder = encoder
            self.decoder = decoder   
        self.z_dim = int(args.latent_dim)
        self.alpha = float(args.latent_alpha)
        self.x_dim=args.encoder_embed_dim
        self.training = True     


        # Moduels
        self.pgn_x_layer = ProbGeneratorNetwork(self.x_dim, len(self.decoder.dictionary))
        self.mlp_xz = nn.Sequential(nn.Linear(self.z_dim, self.x_dim))
        self.mlp_yz = nn.Sequential(nn.Linear(self.z_dim, self.x_dim))

        self.rouge_predict_net =RougePredictNetwork(self.x_dim, 256)
        
        self.XZtoV = nn.Linear(self.x_dim+self.z_dim, len(self.decoder.dictionary), bias=False)
        self.XtoV  = nn.Linear(self.x_dim, len(self.decoder.dictionary), bias=False)
        self.ZtoV  = nn.Linear(self.z_dim, len(self.decoder.dictionary), bias=False)

        nn.init.normal_(self.XZtoV.weight, mean=0, std=1024 ** -0.5)
        nn.init.normal_(self.XtoV.weight, mean=0, std=1024 ** -0.5)
        nn.init.normal_(self.ZtoV.weight, mean=0, std=1024 ** -0.5)
        
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

        xx = encoder_out['encoder_out'][0]
        alpha = self.rouge_predict_net(xx) # -> Dim [B, 1]
        alpha = alpha.unsqueeze(dim=1)     # -> Dim [B, 1, 1]


        x_z= self.pgn_x_layer(xx)

        y, extra = self.decoder(         
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=self.new_linear,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        if self.new_linear:
            y = self.XtoV(y)

        #---- Add it to y
        x_z = alpha*x_z

        y[:,[0],:] += alpha*x_z

        eos: int = self.eos
        if classification_head_name is not None:
            sentence_representation = y[
                src_tokens.eq(eos), :
            ].view(y.size(0), -1, y.size(-1))[:, -1, :]
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    y = head(sentence_representation)
                    break

        extra['x_z']  = x_z

    
        return y, extra                            


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

    def get_latent_z(self, x):
        x_z = self.pgn_x_layer(x)
        return x_z       
    
    def get_latent_vocab(self, z):
        return self.ZtoV(z)

    def get_x_vocab(self, x):
        return self.XtoV(x)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        
        if args.pretrained:
            trained_model = checkpoint_utils.load_model_ensemble(
                filenames=[args.pretrained_checkpoint],
                task=task,
            )[0][0]
            trained_decoder = copy.deepcopy(list(trained_model.children())[1])
            trained_encoder = copy.deepcopy(list(trained_model.children())[0])

            # freeze pretrained model
            for param in trained_decoder.parameters():
                param.requires_grad = not cls.freeze_decoder
            for param in trained_encoder.parameters():
                param.requires_grad = not cls.freeze_encoder
            args.pretrained=False
            print("pretrained model is used")
        
            return cls(args, encoder, decoder, trained_encoder, trained_decoder)
        return cls(args, encoder, decoder)



@register_model_architecture("BAG", "BAG_large")
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

@register_model_architecture("BAG","BAG_base" )
def CVAE_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    CVAE_large_architecture(args)
