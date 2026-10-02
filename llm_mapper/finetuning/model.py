"""Differentiable shared encoder and cosine InfoNCE with explicit negatives."""
import torch
from torch import nn
from torch.nn import functional as F
from ..embedders import create_embedder,EmbeddingConfig
from ..embedders.transformer import pool_hidden
from ..alignment.smith_waterman import reverse_complement


class ContrastiveEncoder(nn.Module):
    def __init__(self,config,device='mps',chunk_length=3998,checkpointing=True):
        super().__init__()
        if config.backend not in {'gb','hf'}:raise ValueError('Differentiable training currently supports GB/HF Transformer backends')
        self.adapter=create_embedder(config,device);self.model=self.adapter.model
        self.tokenizer=self.adapter.tokenizer;self.device=self.adapter.device;self.chunk_length=chunk_length
        if chunk_length>config.max_length-2:raise ValueError('Chunk payload exceeds configured context')
        self.context_checkpointing=checkpointing
        self.model.config.use_cache=False
        if checkpointing:self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        # The unused BERT pooler is not part of nucleotide mean pooling.
        pooler=getattr(self.model,'pooler',None)
        if pooler is not None:
            for p in pooler.parameters():p.requires_grad_(False)
    def encode_raw(self,sequences):
        # Bound retained activations across whole-CDS contexts, not only layers.
        if self.context_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(lambda: self._encode_raw(sequences),use_reentrant=False)
        return self._encode_raw(sequences)

    def _encode_raw(self,sequences):
        strings=sequences
        if self.adapter.config.kmer:
            k=self.adapter.config.kmer;strings=[' '.join(s[i:i+k] for i in range(len(s)-k+1)) for s in strings]
        tokens=self.tokenizer(strings,padding=True,truncation=False,return_special_tokens_mask=True,return_tensors='pt')
        if tokens['input_ids'].shape[1]>self.adapter.context_length:raise ValueError('Training sequence exceeds model context')
        special=tokens.pop('special_tokens_mask').to(self.device);tokens={k:v.to(self.device) for k,v in tokens.items()}
        hidden=self.model(**tokens,use_cache=False).last_hidden_state
        return pool_hidden(hidden,tokens['attention_mask'],special,self.adapter.config.pooling)
    def encode_genes(self,sequences):
        # Separate context calls avoid padding long and short genes together and
        # retain gradients through every context; never silently truncate genes.
        result=[]
        for sequence in sequences:
            chunks=[sequence[i:i+self.chunk_length] for i in range(0,len(sequence),self.chunk_length)]
            values=[self.encode_raw([chunk])[0]*len(chunk) for chunk in chunks]
            result.append(torch.stack(values).sum(dim=0)/len(sequence))
        return F.normalize(torch.stack(result),dim=-1)
    def forward(self,batch):
        anchors=self.encode_genes(batch['anchors'])
        fragments=F.normalize(self.encode_raw(batch['positives']+batch['negatives']),dim=-1)
        return anchors,fragments
    def save_encoder(self,directory):
        # SDPA wraps existing attention modules; export canonical HF weight names
        # so inference can reload through the normal SequenceEmbedder factory.
        state={k.replace('.attention.self.original.','.attention.self.'):v.detach().cpu() for k,v in self.model.state_dict().items()}
        self.model.save_pretrained(directory,state_dict=state,safe_serialization=True)
        self.tokenizer.save_pretrained(directory)


def negative_mask(batch):
    """Exclude same-gene/known-exact positive fragments from the negative set."""
    b=len(batch['anchors']);mask=torch.ones((b,2*b),dtype=torch.bool)
    fragments=batch['positives']+batch['negatives'];donors=batch['anchor_ids']+batch['negative_ids']
    for i,gene in enumerate(batch['anchors']):
        for j,(fragment,donor) in enumerate(zip(fragments,donors)):
            if j!=i and (donor==batch['anchor_ids'][i] or fragment in gene or reverse_complement(fragment) in gene):mask[i,j]=False
    return mask


def info_nce(anchors,fragments,mask,temperature=.05):
    if temperature<=0:raise ValueError('Temperature must be positive')
    b=len(anchors)
    if fragments.shape[0]!=2*b or mask.shape!=(b,2*b):raise ValueError('Need one positive and one explicit negative per anchor')
    labels=torch.arange(b,device=anchors.device);mask=mask.to(anchors.device)
    if not mask[labels,labels].all() or not (mask.sum(dim=1)>=2).all():raise ValueError('Each anchor needs its positive and at least one negative')
    logits=(F.normalize(anchors,dim=-1)@F.normalize(fragments,dim=-1).T)/temperature
    logits=logits.masked_fill(~mask,float('-inf'))
    return F.cross_entropy(logits,labels)
