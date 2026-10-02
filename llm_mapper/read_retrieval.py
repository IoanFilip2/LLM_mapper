"""Zero-shot short-read retrieval against fixed gene vectors, without fitting labels."""
import argparse
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .gene_database import GeneVectorDatabase
from .gene_umap import GENES_OF_INTEREST,COLORS
from .gene_clusters import write_csv
from .index import normalize
from .preparation.downloads import file_hash,write_json
from .alignment.smith_waterman import reverse_complement


class IntervalLabels:
    def __init__(self,blocks):
        self.chromosomes={}
        for chrom in {b['chrom'] for b in blocks}:
            rows=sorted((b['genomic_start'],b['genomic_end'],b['gene_symbol']) for b in blocks if b['chrom']==chrom)
            self.chromosomes[chrom]=(rows,[r[0] for r in rows],np.maximum.accumulate([r[1] for r in rows]).tolist())
    def overlaps(self,chrom,start,end):
        rows,starts,maxends=self.chromosomes[chrom]
        lo=bisect_right(maxends,start);hi=bisect_left(starts,end)
        return {g for a,b,g in rows[lo:hi] if b>start}


def exclusion(read,length=100,mapq=30):
    if read.is_unmapped:return 'unmapped'
    if read.is_secondary or read.is_supplementary:return 'nonprimary'
    if read.is_duplicate:return 'duplicate'
    if read.is_qcfail:return 'vendor_qc_fail'
    if read.mapping_quality==255 or read.mapping_quality<mapq:return 'low_or_unavailable_mapq'
    if read.cigartuples!=[(0,length)] or read.query_length!=length:return 'not_exact_100M'
    if not read.has_tag('AS'):return 'missing_AS'
    if read.query_sequence is None or set(read.query_sequence.upper())-set('ACGT'):return 'missing_or_ambiguous_bases'
    if read.query_qualities is None:return 'missing_base_qualities'
    return None


def select_reads(cram,decoding_reference,cds_blocks,output,targets=None,per_gene=100,seed=20260930,mapq=30):
    import pysam
    targets=targets or GENES_OF_INTEREST
    if per_gene<1 or mapq<0:raise ValueError('Invalid sample size or MAPQ')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    blocks=[json.loads(line) for line in Path(cds_blocks).read_text().splitlines()]
    lookup=IntervalLabels(blocks);selected=[];summaries=[]
    signature=dict(cram_sha256=file_hash(cram),cram_index_sha256=file_hash(str(cram)+'.crai'),
                   cds_blocks_sha256=file_hash(cds_blocks),decoding_reference_sha256=file_hash(decoding_reference),
                   targets=targets,per_gene=per_gene,seed=seed,mapq=mapq,read_length=100,
                   rules='primary, nonduplicate, vendor-QC pass, MAPQ>=threshold and !=255, CIGAR 100M, ACGT, qualities present, fully contained selected CDS, no other selected-gene CDS overlap; maximum AS among eligible reads per gene')
    manifest_path=output/'selection_manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['signature']!=signature:raise ValueError('Selection provenance changed')
        if file_hash(output/'reads.jsonl')!=previous['reads_sha256']:raise ValueError('Selected reads checksum mismatch')
        return [json.loads(line) for line in (output/'reads.jsonl').read_text().splitlines()]
    with pysam.AlignmentFile(str(cram),'rc',reference_filename=str(decoding_reference)) as handle:
        if not handle.has_index():raise ValueError('Indexed CRAM required')
        samples={r.get('SM') for r in handle.header.to_dict().get('RG',[])}
        for gene_index,gene in enumerate(targets):
            gene_blocks=[b for b in blocks if b['gene_symbol']==gene]
            if not gene_blocks:raise ValueError(f'Missing CDS intervals for {gene}')
            rng=np.random.default_rng(np.random.SeedSequence([seed,gene_index]));seen=set();counters=Counter();hist=Counter()
            best=None;tier_count=0;reservoir=[]
            for block in gene_blocks:
                chrom=block['chrom'];contig=chrom if chrom in handle.references else 'chr'+chrom
                if contig not in handle.references:raise ValueError(f'Missing chromosome {chrom}')
                for read in handle.fetch(contig,block['genomic_start'],block['genomic_end']):
                    counters['fetched_region_records']+=1;reason=exclusion(read,mapq=mapq)
                    if reason:counters[reason]+=1;continue
                    start,end=read.reference_start,read.reference_end
                    if start<block['genomic_start'] or end>block['genomic_end']:
                        counters['not_fully_contained_in_CDS_block']+=1;continue
                    if lookup.overlaps(chrom,start,end)!={gene}:
                        counters['ambiguous_CDS_gene_overlap']+=1;continue
                    group=read.get_tag('RG') if read.has_tag('RG') else ''
                    mate=1 if read.is_read1 else 2 if read.is_read2 else 0
                    key=(group,read.query_name,mate)
                    if key in seen:counters['repeated_identity']+=1;continue
                    seen.add(key);counters['eligible_reads']+=1;score=int(read.get_tag('AS'));hist[score]+=1
                    if best is None or score>best:best=score;tier_count=0;reservoir=[]
                    if score<best:continue
                    tier_count+=1
                    slot=len(reservoir) if len(reservoir)<per_gene else int(rng.integers(tier_count))
                    if slot>=per_gene:continue
                    sequence=read.get_forward_sequence().upper()
                    record=dict(read_id=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24],
                        query_name=read.query_name,read_group=group,mate=mate,true_gene=gene,sequence=sequence,
                        contig=contig,reference_start=start,reference_end=end,cigar=read.cigarstring,
                        alignment_score=score,mapq=read.mapping_quality,nm=int(read.get_tag('NM')) if read.has_tag('NM') else None,
                        mean_base_quality=float(np.mean(read.query_qualities)),alignment_reverse=read.is_reverse,
                        cds_strand=block['strand'],transcript_id=block['transcript_id'])
                    if slot==len(reservoir):reservoir.append(record)
                    else:reservoir[slot]=record
            reservoir.sort(key=lambda r:r['read_id']);selected.extend(reservoir)
            summaries.append(dict(gene_symbol=gene,highest_eligible_AS=best,eligible_reads=counters['eligible_reads'],
                           highest_AS_tier_reads=tier_count,selected_reads=len(reservoir),AS_histogram=dict(hist),filter_counts=dict(counters)))
            print(f'{gene}: eligible={counters["eligible_reads"]}, highest AS={best}, tier={tier_count}, selected={len(reservoir)}',flush=True)
    if len({r['read_id'] for r in selected})!=len(selected):raise ValueError('Read selected for multiple genes or identity collision')
    (output/'reads.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    (output/'reads.fa').write_text(''.join(f'>{r["read_id"]}\n{r["sequence"]}\n' for r in selected))
    write_csv(output/'selection_counts.csv',[{k:v for k,v in r.items() if k not in ['AS_histogram','filter_counts']} for r in summaries])
    write_json(manifest_path,dict(status='complete',signature=signature,samples=sorted(samples),genes=summaries,
              reads=len(selected),reads_sha256=file_hash(output/'reads.jsonl'),
              orientation='Original sequenced orientation; true gene strand never used for embedding/retrieval',
              sampling='Uniform reservoir sample within highest eligible AS tier; seed fixed per target gene'))
    return selected


def classification_metrics(truth,predicted,targets):
    if len(truth)!=len(predicted) or not len(truth):raise ValueError('Nonempty matching labels required')
    truth=np.asarray(truth);predicted=np.asarray(predicted);rows=[]
    for gene in targets:
        yes=truth==gene;called=predicted==gene
        tp=int(sum(yes&called));fn=int(sum(yes&~called));fp=int(sum(~yes&called));tn=int(sum(~yes&~called))
        def ratio(a,b):return a/b if b else None
        rows.append(dict(gene_symbol=gene,support=int(sum(yes)),tp=tp,fn=fn,fp=fp,tn=tn,
                         sensitivity=ratio(tp,tp+fn),specificity=ratio(tn,tn+fp),precision=ratio(tp,tp+fp),
                         f1=ratio(2*tp,2*tp+fp+fn)))
    recalls=[r['sensitivity'] for r in rows if r['sensitivity'] is not None]
    return dict(reads=len(truth),correct=int(sum(truth==predicted)),accuracy=float(np.mean(truth==predicted)),
                macro_sensitivity=float(np.mean(recalls)),per_gene=rows,
                specificity_scope='One-versus-rest among sampled target-gene reads only; not genome-wide specificity')


def cosine_retrieve(query_forward,query_reverse,references,candidate_indices):
    candidates=np.asarray(candidate_indices,dtype=int)
    forward=query_forward@references[candidates].T;reverse=query_reverse@references[candidates].T
    scores=np.maximum(forward,reverse)
    # Candidate order is database order, providing deterministic ties.
    order=np.argsort(-scores,axis=1,kind='stable');best=order[:,0];row=np.arange(len(scores))
    return dict(predicted=candidates[best],scores=scores,ranking=candidates[order],
                cosine=scores[row,best],orientation=np.where(reverse[row,best]>forward[row,best],'reverse_complement','forward'),
                margin=scores[row,best]-scores[row,order[:,1]] if len(candidates)>1 else np.full(len(row),np.nan))


def embed_reads(database,selection,output,device='mps',batch_size=32):
    from .embedders import EmbeddingConfig,create_embedder
    db=GeneVectorDatabase(database);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reads=[json.loads(line) for line in (Path(selection)/'reads.jsonl').read_text().splitlines()]
    if not reads:raise ValueError('No selected reads')
    config=EmbeddingConfig(**db.manifest['signature']['model_config'])
    signature=dict(model_fingerprint=db.manifest['signature']['model_fingerprint'],reads_sha256=file_hash(Path(selection)/'reads.jsonl'),
                   orientations=['forward','reverse_complement'],normalization='L2; same raw model pooling as reference')
    path=output/'embedding_manifest.json'
    if path.exists():
        m=json.loads(path.read_text())
        if m['signature']!=signature:raise ValueError('Read embedding signature mismatch')
        if m['status']=='complete' and file_hash(output/'read_vectors.npy')==m['vectors_sha256']:return
        raise ValueError('Incomplete or corrupt read vectors; use a new output directory')
    embedder=create_embedder(config,device)
    if embedder.fingerprint!=signature['model_fingerprint']:raise ValueError('Read and reference model fingerprints differ')
    sequences=[r['sequence'] for r in reads];oriented=sequences+[reverse_complement(s) for s in sequences]
    start=time.perf_counter();vectors=[]
    for offset in range(0,len(oriented),batch_size):
        vectors.append(normalize(embedder.encode(oriented[offset:offset+batch_size],batch_size=batch_size)))
        print(f'Embedded {min(offset+batch_size,len(oriented))}/{len(oriented)} oriented reads',flush=True)
    matrix=np.concatenate(vectors).reshape(2,len(reads),-1)
    if matrix.shape[2]!=db.manifest['signature']['dimension']:raise ValueError('Dimension mismatch')
    np.save(output/'read_vectors.npy',matrix,allow_pickle=False)
    write_json(path,dict(status='complete',signature=signature,device=embedder.device,model_config=asdict(embedder.config),
               reads=len(reads),seconds=time.perf_counter()-start,batch_size=batch_size,vectors_sha256=file_hash(output/'read_vectors.npy')))


def evaluate(database,selection,embeddings,reference_umap,output,study_label='Zero-shot'):
    from .reference_umap import load_reference,project_queries
    output=Path(output);output.mkdir(parents=True,exist_ok=True);db=GeneVectorDatabase(database)
    selection=Path(selection);embeddings=Path(embeddings)
    sm=json.loads((selection/'selection_manifest.json').read_text());em=json.loads((embeddings/'embedding_manifest.json').read_text())
    if file_hash(selection/'reads.jsonl')!=sm['reads_sha256'] or sm['reads_sha256']!=em['signature']['reads_sha256']:raise ValueError('Read provenance mismatch')
    if em['signature']['model_fingerprint']!=db.manifest['signature']['model_fingerprint']:raise ValueError('Model mismatch')
    if file_hash(embeddings/'read_vectors.npy')!=em['vectors_sha256']:raise ValueError('Read vectors corrupted')
    reads=[json.loads(line) for line in (selection/'reads.jsonl').read_text().splitlines()]
    vectors=np.load(embeddings/'read_vectors.npy',allow_pickle=False);reference=np.load(Path(database)/'gene_vectors.npy',allow_pickle=False)
    symbols=[g['gene_symbol'] for g in db.genes];targets=sm['signature']['targets'];truth=[r['true_gene'] for r in reads]
    indices=[i for i,s in enumerate(symbols) if s in targets]
    if len(indices)!=len(targets):raise ValueError('Target genes missing from reference')
    if vectors.shape!=(2,len(reads),reference.shape[1]) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=2),1,atol=1e-5):raise ValueError('Invalid query vectors')
    results={};predictions=[]
    for scope,candidates in [('all_hgnc',np.arange(len(symbols))),('targets_only',indices)]:
        result=cosine_retrieve(vectors[0],vectors[1],reference,candidates);results[scope]=result
        predicted=[symbols[i] for i in result['predicted']];metrics=classification_metrics(truth,predicted,targets)
        truth_indices=np.array([symbols.index(g) for g in truth]);ranks=np.argmax(result['ranking']==truth_indices[:,None],axis=1)+1
        metrics.update(candidate_genes=len(candidates),top5_accuracy=float(np.mean(ranks<=5)),top50_accuracy=float(np.mean(ranks<=50)),top100_accuracy=float(np.mean(ranks<=100)),top250_accuracy=float(np.mean(ranks<=250)),mean_reciprocal_rank=float(np.mean(1/ranks)))
        folder=output/scope;folder.mkdir(exist_ok=True);write_json(folder/'metrics.json',metrics);write_csv(folder/'per_gene_metrics.csv',metrics['per_gene'])
        confusion=Counter(zip(truth,predicted));write_csv(folder/'confusion_long.csv',[dict(true_gene=t,predicted_gene=p,reads=n) for (t,p),n in sorted(confusion.items())])
        rows=[]
        for i,r in enumerate(reads):
            rows.append(dict(read_id=r['read_id'],true_gene=truth[i],predicted_gene=predicted[i],correct=predicted[i]==truth[i],
                        cosine=float(result['cosine'][i]),margin=float(result['margin'][i]),orientation=result['orientation'][i],true_gene_rank=int(ranks[i]),
                        top5=';'.join(symbols[j] for j in result['ranking'][i,:5])))
        write_csv(folder/'predictions.csv',rows)
        report=[f'# {study_label} read retrieval: {scope}','',f'{len(reads)} reads; {len(candidates):,} callable gene labels.',
                f'Accuracy: {metrics["accuracy"]:.2%}; macro sensitivity: {metrics["macro_sensitivity"]:.2%}; top-5 accuracy: {metrics["top5_accuracy"]:.2%}; top-50 accuracy: {metrics["top50_accuracy"]:.2%}; top-100 accuracy: {metrics["top100_accuracy"]:.2%}; top-250 accuracy: {metrics["top250_accuracy"]:.2%}.','',
                '| Gene | Reads | Sensitivity | Specificity | Precision | F1 |','|---|---:|---:|---:|---:|---:|']
        if len(candidates)<=50:
            report.insert(4,'Top-50, top-100 and top-250 include every candidate in this universe and are therefore trivially 100%.')
        for r in metrics['per_gene']:
            values=[f'{r[k]:.3f}' if r[k] is not None else 'NA' for k in ['sensitivity','specificity','precision','f1']]
            report.append(f'| {r["gene_symbol"]} | {r["support"]} | '+ ' | '.join(values)+' |')
        report+=['','Specificity is conditional on the other sampled target genes, not all genomic origins.',
                 'Labels are alignment-derived, not independent biological truth. Reads are selected from the highest eligible AS tier per gene.',
                 'Both orientations are encoded; the larger cosine per candidate is used without consulting truth strand. Ties use reference database order.',
                 'Individual mates are read-level observations; sequence-identical reads and shared templates need not be independent.',
                 'Reference vectors summarize complete CDS contexts, whereas queries are 100 bp; length mismatch may affect retrieval. No retrieval classifier or label fitting; encoder training status is given by the study label.']
        (folder/'report.md').write_text('\n'.join(report)+'\n')
    model=load_reference(reference_umap,file_hash(Path(database)/'gene_vectors.npy'),symbols)
    # Transform every query in both orientations once, never fit or refit on queries.
    xy=project_queries(model,vectors.reshape(-1,vectors.shape[-1])).reshape(2,len(reads),2)
    np.save(output/'query_umap_coordinates.npy',xy,allow_pickle=False)
    chosen=np.where(results['all_hgnc']['orientation']=='reverse_complement',1,0)
    write_csv(output/'query_umap_coordinates.csv',[dict(read_id=r['read_id'],true_gene=r['true_gene'],
              forward_UMAP1=float(xy[0,i,0]),forward_UMAP2=float(xy[0,i,1]),reverse_UMAP1=float(xy[1,i,0]),reverse_UMAP2=float(xy[1,i,1]),
              displayed_orientation=results['all_hgnc']['orientation'][i]) for i,r in enumerate(reads)])
    plot_queries(output,model.embedding_,xy[chosen,np.arange(len(reads))],symbols,truth,targets,study_label)
    plot_confusions(output,truth,results,symbols,targets)
    write_json(output/'manifest.json',dict(status='complete',study_label=study_label,reference_vectors_sha256=file_hash(Path(database)/'gene_vectors.npy'),
               reads_sha256=sm['reads_sha256'],read_vectors_sha256=em['vectors_sha256'],reference_umap_model_sha256=file_hash(Path(reference_umap)/'reference_umap.joblib'),
               reference_coordinates_unchanged=True,projection='Saved reference-only UMAP.transform; both orientations in one fixed query batch; displayed orientation chosen by all-HGNC retrieval, never truth',
               query_transform_input_sha256=em['vectors_sha256'],targets=targets,reads=len(reads)))


def plot_queries(output,reference_xy,query_xy,symbols,truth,targets,study_label='Zero-shot'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    fig,ax=plt.subplots(figsize=(13,9),layout='constrained');ax.scatter(*reference_xy.T,s=5,c='#c5cbd2',alpha=.35,rasterized=True,linewidths=0)
    texts=[]
    for gene,color in zip(targets,COLORS):
        mask=np.array(truth)==gene
        ax.scatter(*query_xy[mask].T,s=17,c=color,alpha=.7,label=f'{gene} (n={sum(mask)})',rasterized=True,linewidths=0)
        position=reference_xy[symbols.index(gene)];ax.scatter(*position,s=140,marker='*',c=color,edgecolors='black',linewidths=.5,zorder=5)
        texts.append(ax.text(*position,gene,color=color,weight='bold',fontsize=9))
    np.random.seed(42);adjust_text(texts,ax=ax,iter_lim=200,arrowprops=dict(arrowstyle='-',color='gray',lw=.5))
    ax.set(title=f'{study_label} WES read retrieval · fixed reference-only UMAP\nDots: reads colored by alignment-derived origin; stars: reference genes',xlabel='UMAP 1',ylabel='UMAP 2',xticks=[],yticks=[])
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
    for ext in ['png','pdf','svg']:fig.savefig(output/f'query_reference_umap.{ext}',dpi=240)
    plt.close(fig)


def plot_confusions(output,truth,results,symbols,targets):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(17,7),layout='constrained')
    for ax,(scope,result) in zip(axes,results.items()):
        labels=targets+(['Other HGNC'] if scope=='all_hgnc' else [])
        matrix=np.zeros((len(targets),len(labels)),int)
        for actual,index in zip(truth,result['predicted']):
            predicted=symbols[index];matrix[targets.index(actual),labels.index(predicted) if predicted in targets else -1]+=1
        image=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(1,matrix.max()));fig.colorbar(image,ax=ax,shrink=.6,label='Reads')
        ax.set(xticks=range(len(labels)),xticklabels=labels,yticks=range(len(targets)),yticklabels=targets,title=scope,xlabel='Predicted gene',ylabel='True gene')
        plt.setp(ax.get_xticklabels(),rotation=60,ha='right')
        for (i,j),count in np.ndenumerate(matrix):
            if count:ax.text(j,i,str(count),ha='center',va='center',fontsize=8,color='white' if count>matrix.max()/2 else 'black')
    for ext in ['png','pdf']:fig.savefig(output/f'confusion_matrices.{ext}',dpi=220)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('select')
    for name in ['cram','decoding_reference','cds_blocks','output']:s.add_argument(name)
    s.add_argument('--per-gene',type=int,default=100);s.add_argument('--mapq',type=int,default=30);s.add_argument('--seed',type=int,default=20260930)
    s=sub.add_parser('embed')
    for name in ['database','selection','output']:s.add_argument(name)
    s.add_argument('--device',default='mps');s.add_argument('--batch-size',type=int,default=32)
    s=sub.add_parser('evaluate')
    for name in ['database','selection','embeddings','reference_umap','output']:s.add_argument(name)
    s.add_argument('--study-label',default='Zero-shot')
    args=vars(p.parse_args());command=args.pop('command');{'select':select_reads,'embed':embed_reads,'evaluate':evaluate}[command](**args)


if __name__=='__main__':main()
