"""Deterministic gene-disjoint triplet plans; sequences are sliced on demand."""
import json
from pathlib import Path
import numpy as np
import pysam
from ..gene_database import GeneVectorDatabase
from ..gene_umap import GENES_OF_INTEREST
from ..preparation.downloads import file_hash,write_json
from ..alignment.smith_waterman import reverse_complement

FIELDS=['anchor','positive_start','positive_length','negative_gene','negative_start','negative_length','positive_rc','negative_rc']


def sample_fragment(rng,sequence,min_length,max_length,blocks=None):
    if blocks is None:
        length=int(rng.integers(min_length,min(max_length,len(sequence))+1))
        return int(rng.integers(len(sequence)-length+1)),length
    longest=max(b-a for a,b in blocks)
    length=int(rng.integers(min_length,min(max_length,longest)+1))
    counts=np.array([max(0,b-a-length+1) for a,b in blocks])
    ticket=int(rng.integers(int(counts.sum())))
    for (start,end),count in zip(blocks,counts):
        if ticket<count:return start+ticket,length
        ticket-=int(count)
    raise AssertionError('Invalid exon sampling ticket')


def sample_triplet(rng,anchor,donors,sequences,min_length=50,max_length=250,reverse_augment=True,regions=None):
    gene=sequences[anchor]
    start,length=sample_fragment(rng,gene,min_length,max_length,None if regions is None else regions[anchor])
    for _ in range(1000):
        negative=int(rng.choice(donors))
        if negative==anchor:continue
        donor=sequences[negative];ns,nl=sample_fragment(rng,donor,min_length,max_length,None if regions is None else regions[negative])
        fragment=donor[ns:ns+nl]
        if fragment not in gene and reverse_complement(fragment) not in gene:break
    else:raise ValueError('Cannot find an unambiguous random negative')
    return [anchor,start,length,negative,ns,nl,int(reverse_augment and rng.random()<.5),int(reverse_augment and rng.random()<.5)]


def prepare(database,fasta,output,train_genes=1000,validation_genes=100,pairs_per_gene=200,
            validation_pairs=20,seed=42,min_length=50,max_length=250,reverse_augment=True,cds_blocks=None):
    if train_genes<2 or validation_genes<2 or min(pairs_per_gene,validation_pairs,min_length)<1 or max_length<min_length:
        raise ValueError('Invalid split or substring configuration')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a new dataset directory; datasets are immutable')
    db=GeneVectorDatabase(database)
    if file_hash(fasta)!=db.manifest['signature']['fasta_sha256']:raise ValueError('Training FASTA differs from base reference')
    eligible=sorted(g['gene_symbol'] for g in db.genes if g['gene_symbol'] not in GENES_OF_INTEREST and g['sequence_length']>=min_length)
    by_gene={}
    if cds_blocks:
        catalog={g['gene_symbol']:g for g in db.genes}
        for line in Path(cds_blocks).read_text().splitlines():
            block=json.loads(line);gene=block['gene_symbol']
            if gene not in catalog:continue
            if block['transcript_id']!=catalog[gene]['transcript_id']:raise ValueError('CDS transcript differs from reference')
            a,b=int(block['cds_start']),int(block['cds_end'])
            if not 0<=a<b<=catalog[gene]['sequence_length']:raise ValueError('Invalid CDS block bounds')
            if b-a!=block['genomic_end']-block['genomic_start']:raise ValueError('CDS/genomic block lengths differ')
            by_gene.setdefault(gene,[]).append((a,b))
        for gene,blocks in by_gene.items():
            blocks.sort()
            if blocks[0][0]!=0 or blocks[-1][1]!=catalog[gene]['sequence_length'] or any(x[1]!=y[0] for x,y in zip(blocks,blocks[1:])):
                raise ValueError('CDS blocks do not tile selected transcript')
        eligible=[g for g in eligible if g in by_gene and max(b-a for a,b in by_gene[g])>=min_length]
    if len(eligible)<train_genes+validation_genes:raise ValueError('Not enough eligible genes')
    rng=np.random.default_rng(seed);chosen=rng.choice(eligible,size=train_genes+validation_genes,replace=False).tolist()
    with pysam.FastaFile(str(fasta)) as f:sequences=[f.fetch(g) for g in chosen]
    if any(set(s)-set('ACGT') for s in sequences):raise ValueError('Training currently requires unambiguous ACGT reference')
    regions=[by_gene[g] for g in chosen] if cds_blocks else None
    output.mkdir(parents=True,exist_ok=True)
    records=[dict(index=i,gene_symbol=g,split='train' if i<train_genes else 'validation',sequence=s,cds_blocks=regions[i] if regions else None) for i,(g,s) in enumerate(zip(chosen,sequences))]
    (output/'genes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    artifacts={};counts={}
    for split,ids,count in [('train',np.arange(train_genes),pairs_per_gene),('validation',np.arange(train_genes,len(chosen)),validation_pairs)]:
        values=np.array([sample_triplet(rng,int(i),ids,sequences,min_length,max_length,reverse_augment,regions) for i in ids for _ in range(count)],dtype=np.int32)
        np.save(output/f'{split}_triplets.npy',values,allow_pickle=False)
        artifacts[f'{split}_triplets.npy']=file_hash(output/f'{split}_triplets.npy');counts[split]=dict(genes=len(ids),triplets=len(values),positive_pairs=len(values),negative_pairs=len(values))
    artifacts['genes.jsonl']=file_hash(output/'genes.jsonl')
    lengths=np.array([len(s) for s in sequences[:train_genes]])
    manifest=dict(status='complete',format_version=1,seed=seed,counts=counts,fields=FIELDS,
                  min_length=min_length,max_length=max_length,reverse_complement_augmentation=reverse_augment,
                  held_out_genes=GENES_OF_INTEREST,donor_policy='Same split as anchor, different gene, reject exact fragment or reverse-complement occurrence in anchor',
                  short_gene_policy='Uniform length from min_length through min(max_length, CDS length); shorter genes excluded',
                  positive_policy='Uniform start in complete spliced CDS; exon-junction-spanning substrings are allowed',
                  fragment_scope='single_CDS_block' if cds_blocks else 'spliced_CDS',
                  cds_blocks_sha256=file_hash(cds_blocks) if cds_blocks else None,
                  source_fasta_sha256=db.manifest['signature']['fasta_sha256'],base_model_config=db.manifest['signature']['model_config'],
                  base_model_fingerprint=db.manifest['signature']['model_fingerprint'],chunk_length=db.manifest['signature']['chunk_length'],
                  train_length_quantiles=dict(zip(['min','p25','median','p75','p95','p99','max'],np.quantile(lengths,[0,.25,.5,.75,.95,.99,1]).tolist())),
                  train_contexts=int(np.ceil(lengths/db.manifest['signature']['chunk_length']).sum()),artifacts=artifacts)
    if cds_blocks:
        manifest['positive_policy']='Uniform feasible length, then uniform valid start across single CDS blocks; applies to positive and negative fragments'
        manifest['short_gene_policy']='Require a CDS block >= min_length; cap length at longest block or max_length, whichever is smaller'
    write_json(output/'manifest.json',manifest);return manifest


class TripletData:
    def __init__(self,directory,split='train'):
        root=Path(directory);self.manifest=json.loads((root/'manifest.json').read_text())
        for name,digest in self.manifest['artifacts'].items():
            if file_hash(root/name)!=digest:raise ValueError('Dataset checksum mismatch')
        self.genes=[json.loads(x) for x in (root/'genes.jsonl').read_text().splitlines()]
        self.pairs=np.load(root/f'{split}_triplets.npy',allow_pickle=False)
        self.signature=file_hash(root/'manifest.json')
    def batch(self,indices):
        anchors=[];positive=[];negative=[];ids=[];donors=[]
        for index in indices:
            a,ps,pl,n,ns,nl,pr,nr=map(int,self.pairs[index]);gene=self.genes[a]['sequence'];pos=gene[ps:ps+pl];neg=self.genes[n]['sequence'][ns:ns+nl]
            anchors.append(gene);positive.append(reverse_complement(pos) if pr else pos);negative.append(reverse_complement(neg) if nr else neg);ids.append(a);donors.append(n)
        return dict(anchors=anchors,positives=positive,negatives=negative,anchor_ids=ids,negative_ids=donors)
