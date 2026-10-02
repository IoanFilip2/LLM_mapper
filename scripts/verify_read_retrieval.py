"""Independent sequence, retrieval, and full-CDS positive controls for the read study."""
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
import pysam
from llm_mapper.gene_database import GeneVectorDatabase
from llm_mapper.embedders import EmbeddingConfig,create_embedder
from llm_mapper.index import normalize
from llm_mapper.alignment.smith_waterman import reverse_complement
from llm_mapper.preparation.downloads import file_hash,write_json
from llm_mapper.read_retrieval import classification_metrics


def verify(database,study,cds_fasta,cds_blocks,device='cpu'):
    study=Path(study);db=GeneVectorDatabase(database);symbols=[g['gene_symbol'] for g in db.genes]
    if file_hash(cds_fasta)!=db.manifest['signature']['fasta_sha256']:raise ValueError('CDS reference differs from database')
    reads=[json.loads(x) for x in (study/'selection/reads.jsonl').read_text().splitlines()]
    blocks=[json.loads(x) for x in Path(cds_blocks).read_text().splitlines()]
    by_gene={g:[b for b in blocks if b['gene_symbol']==g] for g in {r['true_gene'] for r in reads}}
    exact=0
    with pysam.FastaFile(str(cds_fasta)) as fasta:
        for r in reads:
            matches=[b for b in by_gene[r['true_gene']] if b['genomic_start']<=r['reference_start'] and b['genomic_end']>=r['reference_end']]
            if len(matches)!=1:raise ValueError('Ambiguous CDS offset')
            b=matches[0]
            offset=b['cds_start']+(r['reference_start']-b['genomic_start'] if b['strand']=='+' else b['genomic_end']-r['reference_end'])
            seq=reverse_complement(r['sequence']) if r['alignment_reverse'] else r['sequence']
            if b['strand']=='-':seq=reverse_complement(seq)
            exact+=seq==fasta.fetch(r['true_gene'],offset,offset+100)
        controls=['TP53','KRAS','CDKN2A'];sequences=[fasta.fetch(g) for g in controls]
    qc=dict(reads=len(reads),exact_substrings_of_selected_CDS=exact,unique_read_ids=len({r['read_id'] for r in reads}),
            unique_sequences=len({r['sequence'] for r in reads}),unique_templates=len({(r['read_group'],r['query_name']) for r in reads}),
            minimum_mapq=min(r['mapq'] for r in reads),minimum_mean_base_quality=min(r['mean_base_quality'] for r in reads),
            mean_base_quality_below20=sum(r['mean_base_quality']<20 for r in reads),mean_base_quality_below30=sum(r['mean_base_quality']<30 for r in reads),
            NM_histogram=dict(Counter(r['nm'] for r in reads)),AS_histogram=dict(Counter(r['alignment_score'] for r in reads)))
    # Recompute a few predictions using scalar dot products and both orientations.
    refs=np.load(Path(database)/'gene_vectors.npy',allow_pickle=False);query=np.load(study/'embeddings/read_vectors.npy',allow_pickle=False)
    with (study/'evaluation/all_hgnc/predictions.csv').open() as h:predictions=list(csv.DictReader(h))
    for i in [0,len(reads)//2,len(reads)-1]:
        scores=np.array([max(float(np.dot(query[0,i],ref)),float(np.dot(query[1,i],ref))) for ref in refs])
        if symbols[int(np.argmax(scores))]!=predictions[i]['predicted_gene']:raise AssertionError('Independent retrieval mismatch')
    qc['independent_dot_product_queries_verified']=3
    embedder=create_embedder(EmbeddingConfig(**db.manifest['signature']['model_config']),device)
    if embedder.fingerprint!=db.manifest['signature']['model_fingerprint']:raise ValueError('Positive-control model fingerprint differs')
    if max(map(len,sequences))>db.manifest['signature']['chunk_length']:raise ValueError('Use single-context control genes')
    vectors=normalize(embedder.encode(sequences,batch_size=1));checks=[]
    for gene,vector in zip(controls,vectors):
        score=refs@vector;index=symbols.index(gene);prediction=symbols[int(np.argmax(score))]
        checks.append(dict(gene_symbol=gene,predicted_gene=prediction,cosine_to_saved_vector=float(score[index])))
        if prediction!=gene or score[index]<.9999:raise AssertionError('Full-CDS positive control failed')
    qc['positive_control_device']=device;qc['full_CDS_positive_controls']=checks
    for scope in ['all_hgnc','targets_only']:
        with (study/f'evaluation/{scope}/predictions.csv').open() as h:pred=list(csv.DictReader(h))
        targets=json.loads((study/'selection/selection_manifest.json').read_text())['signature']['targets']
        keep=[i for i,r in enumerate(reads) if r['mean_base_quality']>=30]
        subset=classification_metrics([reads[i]['true_gene'] for i in keep],[pred[i]['predicted_gene'] for i in keep],targets)
        subset['selection_note']='Post-selection mean-base-Q>=30 sensitivity subset of the same sample; no replacement reads or new embeddings'
        write_json(study/f'evaluation/{scope}/q30_subset_metrics.json',subset)
    write_json(study/'verification.json',qc);return qc


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['database','study','cds_fasta','cds_blocks']:p.add_argument(name)
    p.add_argument('--device',default='cpu');print(json.dumps(verify(**vars(p.parse_args())),indent=2))
