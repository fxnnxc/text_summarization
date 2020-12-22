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
from fairseq.modules import MultiheadAttention
import os 
import torch 
import torch.nn as nn
import random 
from typing import Optional


@register_model("pretrain_bart")
class VaeBART(BARTModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        BARTModel.add_args(parser)
        parser.add_argument('--pretrained-checkpoint',
                            help='path to load checkpoint from pretrained model')
        parser.add_argument('--pretrained', action="store_true",
                            help='use pretrained model when training [True, ...]')

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.encoder = encoder 
        self.decoder = decoder  

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
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
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
        return x, extra                               

        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

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

            trained_decoder = list(trained_model.children())[1]
            trained_encoder = list(trained_model.children())[0]
            # freeze pretrained model
            for param in trained_decoder.parameters():
                param.requires_grad = True
            for param in trained_encoder.parameters():
                 param.requires_grad = True
        model = VaeBART(args, trained_encoder, trained_decoder)

        return model 
        


@register_model_architecture("pretrain_bart", "pretrain_bart_large")
def pretrain_bart_large_architecture(args):
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

@register_model_architecture("pretrain_bart","pretrain_bart_base" )
def pretrain_bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    pretrain_bart_large_architecture(args)