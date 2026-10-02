"""Resumable exact Smith-Waterman evaluation on an existing cosine graph."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from importlib.metadata import version
import csv
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr

from .alignment import SmithWaterman, SWConfig
from .gene_database import GeneVectorDatabase
from .gene_clusters import write_csv
from .preparation.downloads import file_hash, write_json


def load_inputs(database, graph_directory, fasta):
    import pysam
    db=GeneVectorDatabase(database);root=Path(graph_directory)
    gm=json.loads((root/'analysis_manifest.json').read_text())
    if gm['status']!='complete':raise ValueError('Graph analysis is incomplete')
    matrix_hash=file_hash(Path(database)/'gene_vectors.npy')
    if matrix_hash!=gm['gene_vectors_sha256']:raise ValueError('Graph and database fingerprints differ')
    if file_hash(fasta)!=db.manifest['signature']['fasta_sha256']:raise ValueError('FASTA differs from embedded reference')
    with (root/'graph_nodes.csv').open() as handle:nodes=list(csv.DictReader(handle))
    if len(nodes)!=len(db.genes) or any(int(r['index'])!=i or r['gene_symbol']!=db.genes[i]['gene_symbol'] for i,r in enumerate(nodes)):
        raise ValueError('Graph node ordering differs from database')
    adjacency=sparse.load_npz(root/'cosine_knn_graph.npz').tocsr()
    if adjacency.shape!=(len(nodes),len(nodes)) or (adjacency-adjacency.T).nnz or adjacency.diagonal().any():
        raise ValueError('Expected symmetric graph with no self loops')
    if not np.isfinite(adjacency.data).all() or (adjacency.data<=0).any():raise ValueError('Graph weights must be positive and finite')
    with pysam.FastaFile(str(fasta)) as reference:sequences=[reference.fetch(g['gene_symbol']) for g in db.genes]
    if any(len(s)!=g['sequence_length'] for s,g in zip(sequences,db.genes)):raise ValueError('Sequence length mismatch')
    with (root/'umap_coordinates.csv').open() as handle:coordinates=list(csv.DictReader(handle))
    by_name={r['gene_symbol']:r for r in coordinates}
    if len(by_name)!=len(nodes) or len(coordinates)!=len(nodes) or set(by_name)!={r['gene_symbol'] for r in nodes}:
        raise ValueError('UMAP labels differ from database')
    xy=np.array([[float(by_name[g['gene_symbol']][k]) for k in ['UMAP1','UMAP2']] for g in db.genes])
    if not np.isfinite(xy).all():raise ValueError('Non-finite UMAP coordinates')
    signature={'gene_vectors_sha256':matrix_hash,'fasta_sha256':db.manifest['signature']['fasta_sha256'],
        'graph_sha256':file_hash(root/'cosine_knn_graph.npz'),'nodes_sha256':file_hash(root/'graph_nodes.csv'),
        'coordinates_sha256':file_hash(root/'umap_coordinates.csv')}
    return db,adjacency,sequences,nodes,xy,signature


def aggregate_neighbors(adjacency, rows, cols, similarities, shorter):
    """Mean over the original adjacency, retaining zero-score edges in denominator."""
    n=adjacency.shape[0]
    sw=sparse.csr_matrix((np.r_[similarities,similarities],(np.r_[rows,cols],np.r_[cols,rows])),shape=(n,n))
    short=sparse.csr_matrix((np.r_[shorter,shorter],(np.r_[rows,cols],np.r_[cols,rows])),shape=(n,n))
    ones=np.ones(n);degree=np.diff(adjacency.indptr)
    denom=np.asarray(adjacency@ones).ravel()
    def divide(numer,denom):return np.divide(numer,denom,out=np.full(n,np.nan),where=denom>0)
    return sw,degree,divide(sw@ones,degree),divide(sw.multiply(adjacency)@ones,denom),divide(short@ones,degree)


def draw(output,xy,values,genes,labels,cosines=None,similarities=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    finite=np.isfinite(values)
    upper=max(float(np.quantile(values[finite],.99)),1e-8) if finite.any() else 1.
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,vmax,title in [(axes[0],1.,'Mean normalized SW similarity · full scale'),
                          (axes[1],upper,'Same scores · contrast view (upper 1% color-clipped)')]:
        ax.scatter(*xy[~finite].T,c='#AAAAAA',s=4,linewidths=0)
        plot=ax.scatter(*xy[finite].T,c=values[finite],s=5,linewidths=0,cmap='magma',vmin=0,vmax=vmax,rasterized=True)
        fig.colorbar(plot,ax=ax,shrink=.8,label='Mean normalized local-alignment similarity',extend='max' if vmax<1 else 'neither')
        ax.set(title=title,xlabel='UMAP 1',ylabel='UMAP 2',xticks=[],yticks=[])
    fig.suptitle('Do cosine neighbors share DNA sequence similarity?',fontsize=17)
    for ext in ['png','pdf','svg']:fig.savefig(output/f'neighbor_sw_umap.{ext}',dpi=250)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    groups=np.unique(labels)
    if finite.any():axes[0].hist(values[finite],bins=60,color='#6A4C93')
    axes[0].set(xlabel='Mean normalized SW',ylabel='Genes',title='Neighborhood sequence similarity')
    present=[c for c in groups if np.any(finite & (labels==c))]
    if present:axes[1].boxplot([values[finite & (labels==c)] for c in present],tick_labels=[str(c) for c in present],showfliers=False)
    axes[1].set(xlabel='Louvain cluster',ylabel='Mean normalized SW',title='By embedding cluster')
    artist=axes[2].scatter([g['gc_fraction'] for g in genes],values,c=np.log10([g['sequence_length'] for g in genes]),s=4,cmap='viridis',rasterized=True,linewidths=0)
    fig.colorbar(artist,ax=axes[2],label='log10 CDS length')
    axes[2].set(xlabel='GC fraction',ylabel='Mean normalized SW',title='Composition / length diagnostic')
    for ext in ['png','pdf']:fig.savefig(output/f'neighbor_sw_qc.{ext}',dpi=220)
    plt.close(fig)
    if cosines is not None and len(cosines):
        fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
        artist=ax.hexbin(cosines,similarities,gridsize=70,bins='log',mincnt=1,cmap='viridis')
        fig.colorbar(artist,ax=ax,label='Graph edges (log count scale)')
        ax.set(xlabel='Embedding cosine similarity',ylabel='Normalized SW similarity',
               title='Cosine versus nucleotide alignment similarity\nEvaluated graph edges only; not all gene pairs')
        for ext in ['png','pdf']:fig.savefig(output/f'edge_cosine_sw.{ext}',dpi=220)
        plt.close(fig)


def run(database,graph_directory,fasta,output,workers=8,batch_size=256,config=None,max_batches=None):
    if workers<1 or batch_size<1:raise ValueError('Workers and checkpoint batch must be positive')
    db,adjacency,sequences,nodes,xy,signature=load_inputs(database,graph_directory,fasta)
    config=config or SWConfig();engine=SmithWaterman(config)
    signature.update(scoring=asdict(config),parasail_python_version=version('parasail'),
                     parasail_c_version=list(engine.parasail.version()),
                     parasail_library_sha256=file_hash(engine.parasail.get_library()))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest_path=output/'manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            if manifest['signature']!=signature:raise ValueError('Resume signature mismatch')
        else:
            if (output/'edge_scores.sqlite').exists():raise FileExistsError('Untracked edge database')
            manifest={'signature':signature,'status':'building','execution_history':[]}
        upper=sparse.triu(adjacency,k=1).tocoo()
        sort=np.lexsort((upper.col,upper.row));rows=upper.row[sort];cols=upper.col[sort];cosines=upper.data[sort]
        lengths=np.array([len(s) for s in sequences],dtype=np.int64);cells=lengths[rows]*lengths[cols]*(2 if config.both_strands else 1)
        with sqlite3.connect(output/'edge_scores.sqlite') as sql:
            sql.execute('PRAGMA synchronous=FULL')
            sql.execute('CREATE TABLE IF NOT EXISTS edges (edge_index INTEGER PRIMARY KEY, source INTEGER, target INTEGER, score INTEGER, similarity REAL, shorter_similarity REAL, orientation TEXT)')
            sql.execute('CREATE TABLE IF NOT EXISTS batches (batch_index INTEGER PRIMARY KEY, edges INTEGER, dp_cells INTEGER, seconds REAL)')
            committed=set()
            for index,i,j,score,sim,short,orientation in sql.execute('SELECT * FROM edges'):
                if not 0<=index<len(rows) or (i,j)!=(int(rows[index]),int(cols[index])) or not 0<=sim<=1 or not 0<=short<=1:
                    raise ValueError('Invalid committed alignment edge')
                expected=2*score/(config.match*(len(sequences[i].replace('N',''))+len(sequences[j].replace('N',''))))
                if abs(expected-sim)>1e-12:raise ValueError('Committed normalization mismatch')
                committed.add(index)
            pending=np.array([i for i in np.argsort(-cells,kind='stable') if int(i) not in committed],dtype=int)
            manifest['execution_history'].append({'workers':workers,'checkpoint_batch':batch_size,'resume_edges':len(committed)})
            manifest.update(status='building',total_edges=len(rows),completed_edges=len(committed),total_dp_cells=int(cells.sum()))
            write_json(manifest_path,manifest)
            batch_index=sql.execute('SELECT COALESCE(MAX(batch_index),-1)+1 FROM batches').fetchone()[0]
            def align(index):
                i,j=int(rows[index]),int(cols[index]);result=engine.compare(sequences[i],sequences[j])
                return int(index),i,j,result['score'],result['similarity'],result['shorter_normalized_similarity'],result['orientation']
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for offset in range(0,len(pending),batch_size):
                    selected=pending[offset:offset+batch_size];started=time.perf_counter()
                    values=list(pool.map(align,selected));seconds=time.perf_counter()-started
                    with sql:
                        sql.executemany('INSERT INTO edges VALUES (?,?,?,?,?,?,?)',values)
                        sql.execute('INSERT INTO batches VALUES (?,?,?,?)',(batch_index,len(values),int(cells[selected].sum()),seconds))
                    batch_index+=1;committed.update(int(i) for i in selected)
                    manifest.update(completed_edges=len(committed),completed_dp_cells=int(cells[list(committed)].sum()))
                    write_json(manifest_path,manifest)
                    print(f'Aligned {len(committed):,}/{len(rows):,} pairs; {100*manifest["completed_dp_cells"]/max(int(cells.sum()),1):.1f}% DP work; batch {seconds:.1f}s',flush=True)
                    if max_batches is not None and offset//batch_size+1>=max_batches:return manifest
            scores=list(sql.execute('SELECT * FROM edges ORDER BY edge_index'))
            if len(scores)!=len(rows):raise ValueError('Alignment coverage incomplete')
            similarities=np.array([x[4] for x in scores]);shorter=np.array([x[5] for x in scores])
            sw,degree,mean,weighted,mean_short=aggregate_neighbors(adjacency,rows,cols,similarities,shorter)
            sparse.save_npz(output/'neighbor_sw_similarity.npz',sw)
            records=[]
            for i,gene in enumerate(db.genes):
                records.append({'gene_symbol':gene['gene_symbol'],'hgnc_id':gene.get('hgnc_id',''),
                    'cluster':nodes[i]['cluster'],'neighbors':int(degree[i]),
                    'mean_sw_similarity':float(mean[i]) if degree[i] else None,
                    'mean_sw_distance':float(1-mean[i]) if degree[i] else None,
                    'cosine_weighted_mean_sw_similarity':float(weighted[i]) if degree[i] else None,
                    'mean_shorter_normalized_sw':float(mean_short[i]) if degree[i] else None,
                    'gc_fraction':gene['gc_fraction'],'cds_length':gene['sequence_length'],
                    'UMAP1':float(xy[i,0]),'UMAP2':float(xy[i,1])})
            write_csv(output/'gene_neighbor_sw.csv',records)
            finite=np.isfinite(mean)
            def correlation(a,b):return float(spearmanr(a,b).statistic) if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None
            qc={'genes':len(db.genes),'pairs':len(scores),'isolates':int((degree==0).sum()),
                'edge_cosine_sw_spearman':correlation(cosines,similarities),
                'gene_sw_gc_spearman':correlation(mean[finite],np.array([g['gc_fraction'] for g in db.genes])[finite]),
                'gene_sw_log_length_spearman':correlation(mean[finite],np.log10(lengths[finite])),
                'reverse_complement_wins':sum(x[6]=='reverse_complement' for x in scores),
                'mean_gene_sw_similarity':float(mean[finite].mean()) if finite.any() else None,
                'gene_mean_sw_quantiles':dict(zip(['min','p25','median','p75','p95','p99','max'],np.quantile(mean[finite],[0,.25,.5,.75,.95,.99,1]).tolist())) if finite.any() else {},
                'alignment_batch_seconds':sql.execute('SELECT SUM(seconds) FROM batches').fetchone()[0]}
            write_json(output/'qc.json',qc)
            labels=np.array([int(n['cluster']) for n in nodes])
            cluster_summaries=[]
            for cluster in np.unique(labels):
                values=mean[(labels==cluster)&finite]
                cluster_summaries.append({'cluster':int(cluster),'genes':int((labels==cluster).sum()),
                    'mean_neighbor_sw':float(values.mean()) if len(values) else None,
                    'median_neighbor_sw':float(np.median(values)) if len(values) else None,
                    'p95_neighbor_sw':float(np.quantile(values,.95)) if len(values) else None})
            write_csv(output/'cluster_sw_summary.csv',cluster_summaries)
            draw(output,xy,mean,db.genes,labels,cosines,similarities)
            manifest.update(status='complete',completed_edges=len(scores),completed_dp_cells=int(cells.sum()),
                aggregation='Unweighted mean over union-symmetrized cosine kNN neighbors; weighted mean supplied separately',
                normalization='2 * best SW score / (match * (ACGT_bases(A)+ACGT_bases(B))); N comparisons score zero; orientations specified in scoring config',
                interpretation='Sequence similarity, not proof of homology or an evolutionary distance; UMAP is display only',
                packages={p:version(p) for p in ['parasail','numpy','scipy','matplotlib','pysam']})
            write_json(manifest_path,manifest)
            print(json.dumps(qc,indent=2),flush=True)
            return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database');parser.add_argument('graph_directory');parser.add_argument('fasta');parser.add_argument('output')
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--algorithm',choices=['striped','scan'],default='striped')
    parser.add_argument('--forward-only',action='store_true');parser.add_argument('--max-batches',type=int,help='Checkpoint after this many batches (useful for timing/resume checks)')
    args=parser.parse_args(argv)
    run(args.database,args.graph_directory,args.fasta,args.output,args.workers,args.batch_size,
        SWConfig(both_strands=not args.forward_only,algorithm=args.algorithm),args.max_batches)


if __name__=='__main__':main()
