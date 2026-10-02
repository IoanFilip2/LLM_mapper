"""Reproducible gene-level cosine UMAP with labeled genes and confound diagnostics."""
import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np
from .gene_database import GeneVectorDatabase
from .preparation.downloads import file_hash,write_json

GENES_OF_INTEREST=['TP53','KRAS','HLA-A','HLA-B','HLA-C','EGFR','MYC','BRCA1','BRCA2','CDKN2A','BRAF','MDM2']
COLORS=['#873E8C','#D45D35','#007F82','#279D9A','#71BDB3','#E89936','#BE395B','#4966A7','#8095CC','#8E6D43','#6C883E','#BD8CBB']


def plot_database(database,output,highlight=None,seed=42,neighbors=30,min_dist=.25,replace=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from adjustText import adjust_text
    import umap
    db=GeneVectorDatabase(database)
    from .verify_gene_database import verify_database
    write_json(Path(database)/'verification.json', verify_database(database))
    output=Path(output)
    if output.exists() and any(output.iterdir()) and not replace:
        raise FileExistsError('Plot output is not empty; use a new directory')
    output.mkdir(parents=True,exist_ok=True)
    vectors=np.load(Path(database)/'gene_vectors.npy',mmap_mode='r',allow_pickle=False)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5):
        raise ValueError('Gene vectors must be finite and normalized')
    n=len(vectors)
    if n<4:raise ValueError('UMAP needs at least four genes')
    highlight=highlight or GENES_OF_INTEREST
    print(f'Fitting cosine UMAP on all {n:,} gene vectors',flush=True)
    parameters={'n_components':2,'n_neighbors':min(neighbors,n-1),'min_dist':min_dist,'metric':'cosine',
                'random_state':seed,'n_jobs':1,'init':'spectral'}
    from .reference_umap import fit_reference
    model,projection=fit_reference(vectors,output,parameters,file_hash(Path(database)/'gene_vectors.npy'),
                                   [g['gene_symbol'] for g in db.genes])
    if projection.shape!=(n,2) or not np.isfinite(projection).all():raise ValueError('Invalid UMAP projection')
    with (output/'gene_umap_coordinates.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['gene_symbol','hgnc_id','transcript_id','cds_length','gc_fraction','UMAP1','UMAP2'])
        writer.writeheader()
        for row,xy in zip(db.genes,projection):
            writer.writerow({'gene_symbol':row['gene_symbol'],'hgnc_id':row.get('hgnc_id',''),
                             'transcript_id':row.get('transcript_id',''),'cds_length':row['sequence_length'],
                             'gc_fraction':row['gc_fraction'],'UMAP1':float(xy[0]),'UMAP2':float(xy[1])})
    by_symbol={r['gene_symbol']:i for i,r in enumerate(db.genes)}
    present=[g for g in highlight if g in by_symbol];missing=[g for g in highlight if g not in by_symbol]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#D4DCE2','axes.labelcolor':'#55616D','text.color':'#253341',
                         'xtick.color':'#7C8994','ytick.color':'#7C8994','svg.fonttype':'none'})
    fig=plt.figure(figsize=(14,9),facecolor='#FAFBFC')
    ax=fig.add_axes([.075,.15,.72,.70],facecolor='#FAFBFC')
    ax.scatter(projection[:,0],projection[:,1],s=7,c='#8C9DAA',alpha=.34,linewidths=0,rasterized=True)
    texts=[];handles=[]
    for i,gene in enumerate(present):
        xy=projection[by_symbol[gene]];color=dict(zip(GENES_OF_INTEREST,COLORS)).get(gene,COLORS[i%len(COLORS)])
        ax.scatter(*xy,s=105,c=color,edgecolors='white',linewidths=1.5,zorder=4)
        texts.append(ax.text(*xy,gene,fontsize=10,fontweight='bold',color=color,zorder=5,
                             bbox={'facecolor':'#FAFBFC','edgecolor':'none','alpha':.9,'pad':1.5}))
        handles.append(Line2D([0],[0],marker='o',linestyle='',color=color,markersize=7,label=gene))
    ax.margins(.10)
    np.random.seed(seed)
    adjust_text(texts,ax=ax,expand=(1.3,1.5),force_text=(.5,.8),
                arrowprops={'arrowstyle':'-','color':'#8997A1','lw':.7},iter_lim=300)
    ax.set_xlabel('UMAP 1',labelpad=10);ax.set_ylabel('UMAP 2',labelpad=10)
    ax.set_xticks([]);ax.set_yticks([])
    ax.legend(handles=handles,title='Highlighted genes',loc='center left',bbox_to_anchor=(1.04,.5),
              frameon=False,labelspacing=1.15,title_fontsize=11)
    config=db.manifest['signature']['model_config']
    model=config['model'].split('/')[-1]
    representation=db.manifest['signature'].get('representation','gene')
    fig.text(.075,.94,'A coding-sequence map of human genes' if representation=='spliced_CDS' else 'A sequence map of human genes',
             fontsize=23,fontweight='bold')
    fig.text(.075,.895,f'{model}  /  {n:,} HGNC genes  /  {db.manifest["device"].upper()} {config["dtype"]}',
             fontsize=12,color='#596A78')
    context=db.manifest['signature']['chunk_length']
    fig.text(.075,.075,f'One vector per gene · contexts ≤ {context:,} bases · cosine UMAP · neighbors {parameters["n_neighbors"]} · seed {seed}',
             fontsize=10,color='#596A78')
    fig.text(.075,.045,'Exploratory projection: distances and clusters in 2D do not establish mapping specificity or shared biological function.',
             fontsize=9,color='#788691')
    for ext in ('png','pdf','svg'):fig.savefig(output/f'gene_umap.{ext}',dpi=300,facecolor=fig.get_facecolor())
    plt.close(fig)
    # Same projection, colored by simple sequence features to reveal potential confounds.
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,values,title in [(axes[0],np.log10([r['sequence_length'] for r in db.genes]),'CDS length (log10 bases)'),
                            (axes[1],[r['gc_fraction'] for r in db.genes],'GC fraction')]:
        artist=ax.scatter(projection[:,0],projection[:,1],c=values,s=5,cmap='viridis',rasterized=True,linewidths=0)
        ax.set_title(title);ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2')
        fig.colorbar(artist,ax=ax,shrink=.8)
    fig.suptitle('Sequence-composition diagnostics · same gene embedding projection',fontsize=14)
    fig.savefig(output/'gene_umap_diagnostics.png',dpi=250);fig.savefig(output/'gene_umap_diagnostics.pdf');plt.close(fig)
    # High-dimensional neighbors are recorded separately from the UMAP coordinates.
    neighbors_original={}
    for gene in present:
        index=by_symbol[gene]
        scores=vectors@vectors[index]
        order=np.argsort(-scores,kind='stable')
        order=order[order!=index][:10]
        neighbors_original[gene]=[{'gene_symbol':db.genes[int(j)]['gene_symbol'],'cosine_similarity':float(scores[j])} for j in order]
    write_json(output/'highlight_neighbors_original_space.json',neighbors_original)
    import hashlib
    from collections import Counter
    digests=Counter(hashlib.sha256(row.tobytes()).hexdigest() for row in vectors)
    rng=np.random.default_rng(seed)
    left=rng.integers(0,n,size=10000);right=(left+rng.integers(1,n,size=10000))%n
    pair_cosines=np.einsum('ij,ij->i',vectors[left],vectors[right])
    geometry={'exact_duplicate_vector_rows_beyond_first':sum(c-1 for c in digests.values()),
              'random_distinct_gene_pair_cosine_quantiles':dict(zip(['min','p05','median','p95','max'],
                   np.quantile(pair_cosines,[0,.05,.5,.95,1]).tolist())),
              'random_pairs':10000,'seed':seed}
    write_json(output/'plot_manifest.json',{'parameters':parameters,'genes':n,'dimension':vectors.shape[1],
               'reference_umap_model_sha256':file_hash(output/'reference_umap.joblib'),
               'fit_scope':'reference genes only; queries use saved transform',
               'highlighted':present,'missing_highlights':missing,'geometry_diagnostics':geometry,'gene_vectors_sha256':file_hash(Path(database)/'gene_vectors.npy'),
               'database_signature':db.manifest['signature'],
               'packages':{p:version(p) for p in ['umap-learn','numpy','scikit-learn','matplotlib','adjustText','numba']},
               'interpretation':'UMAP is visualization only; original-space cosine retrieval is the evaluation space.'})
    print(f'Plots and coordinates saved to {output}',flush=True)
    return projection


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('database');p.add_argument('output');p.add_argument('--highlight',nargs='+')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--neighbors',type=int,default=30);p.add_argument('--min-dist',type=float,default=.25)
    p.add_argument('--replace',action='store_true',help='Regenerate figures using the saved reference-only fit')
    args=p.parse_args(argv)
    plot_database(args.database,args.output,args.highlight,args.seed,args.neighbors,args.min_dist,args.replace)


if __name__=='__main__':main()
