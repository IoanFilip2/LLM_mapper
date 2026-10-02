"""Opt-in PyTorch SDPA for GB's bidirectional RNABert attention.

The original learned projections and rotary implementation are retained. This
adapter is scoped to model instances; it does not alter global PyTorch behavior.
"""
import torch
from torch import nn
from torch.nn import functional as F


class GBSdpaAttention(nn.Module):
    def __init__(self,original):
        super().__init__()
        if original.is_decoder or original.position_embedding_type not in {'rope','absolute'}:
            raise ValueError('SDPA adapter requires a bidirectional GB encoder with rope/absolute positions')
        self.original=original

    def forward(self,hidden_states,attention_mask=None,head_mask=None,encoder_hidden_states=None,
                encoder_attention_mask=None,past_key_value=None,output_attentions=False,rotary_pos_emb=None):
        if head_mask is not None or encoder_hidden_states is not None or past_key_value is not None or output_attentions:
            raise ValueError('GB SDPA supports the encoder without head masks, cache or attention outputs')
        from modelgenerator.huggingface_models.rnabert.modeling_rnabert import apply_rotary_pos_emb
        a=self.original
        q=a.transpose_for_scores(a.query(hidden_states))
        k=a.transpose_for_scores(a.key(hidden_states))
        v=a.transpose_for_scores(a.value(hidden_states))
        if rotary_pos_emb is not None:
            qr,kr=rotary_pos_emb if isinstance(rotary_pos_emb,tuple) else (rotary_pos_emb,rotary_pos_emb)
            q=apply_rotary_pos_emb(q.permute(2,0,1,3).contiguous(),qr).permute(1,2,0,3).contiguous()
            k=apply_rotary_pos_emb(k.permute(2,0,1,3).contiguous(),kr).permute(1,2,0,3).contiguous()
        mask=attention_mask.to(q.dtype) if attention_mask is not None else None
        context=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=a.dropout.p if self.training else 0.,is_causal=False)
        context=context.permute(0,2,1,3).contiguous().view(hidden_states.shape[0],hidden_states.shape[1],a.all_head_size)
        return (context,)


def install_gb_sdpa(model):
    for layer in model.encoder.layer:
        layer.attention.self=GBSdpaAttention(layer.attention.self)
