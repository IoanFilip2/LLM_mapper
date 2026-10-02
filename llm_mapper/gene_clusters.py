"""Louvain clustering and composition QC of a completed gene vector database."""
import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .gene_database import GeneVectorDatabase
from .preparation.downloads import file_hash, write_json
from .verify_gene_database import verify_database


def cosine_graph(vectors, neighbors=30, block_size=256):
    """Exact kNN; union symmetrization, positive cosine weights, no self loops.

    Stable sorting breaks exact similarity ties by database row index.
    Only block_size x n similarities are materialized, never the full n x n.
    """
    n = len(vectors)
    if n < 2 or not 1 <= neighbors < n or block_size < 1:
        raise ValueError('Require 1 <= neighbors < gene count and positive block size')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError('Vectors must be finite and unit normalized')
    sources, targets, weights = [], [], []
    for start in range(0, n, block_size):
        scores = np.clip(vectors[start:start+block_size] @ vectors.T, -1, 1)
        scores[np.arange(len(scores)), np.arange(start, start+len(scores))] = -np.inf
        order = np.argsort(-scores, axis=1, kind='stable')[:, :neighbors]
        values = np.take_along_axis(scores, order, axis=1)
        valid = values > 0
        sources.extend(np.broadcast_to(np.arange(start, start+len(scores))[:, None], order.shape)[valid])
        targets.extend(order[valid]); weights.extend(values[valid])
    directed = sparse.csr_matrix((weights, (sources, targets)), shape=(n, n))
    adjacency = directed.maximum(directed.T)
    return nx.from_scipy_sparse_array(adjacency), adjacency


def partition(graph, resolution=.3, seed=42):
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('Resolution must be finite and positive')
    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph]
    else:
        communities = nx.community.louvain_communities(graph, weight='weight', resolution=resolution, seed=seed)
    communities = sorted(communities, key=lambda c: (-len(c), min(c)))
    labels = np.empty(len(graph), dtype=int)
    for label, members in enumerate(communities):
        labels[list(members)] = label
    return communities, labels


def characterize(graph, vectors, genes, communities, labels, top_n=20):
    records, summaries, representatives = [], [], []
    for label, members in enumerate(communities):
        ids = np.array(sorted(members)); n = len(ids)
        subgraph = graph.subgraph(members)
        summed = vectors[ids].sum(axis=0, dtype=np.float64)
        peer_cosines = (vectors[ids] @ summed - np.einsum('ij,ij->i', vectors[ids], vectors[ids])) / max(n-1, 1)
        strengths = dict(subgraph.degree(weight='weight'))
        degrees = dict(subgraph.degree())
        cluster_rows = []
        for local, index in enumerate(ids):
            gene = genes[index]
            total = graph.degree(int(index), weight='weight')
            cluster_rows.append({'gene_symbol': gene['gene_symbol'], 'hgnc_id': gene.get('hgnc_id', ''),
                'transcript_id': gene.get('transcript_id', ''), 'cluster': label,
                'cluster_size': n, 'internal_cosine_strength': float(strengths[index]),
                'cosine_degree_centrality': float(strengths[index]/max(n-1, 1)),
                'internal_degree': degrees[index], 'total_degree': graph.degree(int(index)),
                'internal_weight_fraction': float(strengths[index]/total) if total else 0.,
                'mean_cosine_to_cluster_peers': float(np.clip(peer_cosines[local], -1, 1)) if n > 1 else None,
                'gc_fraction': float(gene['gc_fraction']), 'cds_length': int(gene['sequence_length'])})
        cluster_rows.sort(key=lambda r: (-r['internal_cosine_strength'],
                          -(r['mean_cosine_to_cluster_peers'] or 0), r['gene_symbol']))
        for rank, row in enumerate(cluster_rows, 1): row['centrality_rank'] = rank
        records.extend(cluster_rows); representatives.extend(cluster_rows[:top_n])
        gc = np.array([r['gc_fraction'] for r in cluster_rows]); lengths = [r['cds_length'] for r in cluster_rows]
        total_strength = sum(graph.degree(i, weight='weight') for i in members)
        summaries.append({'cluster': label, 'genes': n, 'internal_edges': subgraph.number_of_edges(),
            'connected_components': nx.number_connected_components(subgraph),
            'gc_median': float(np.median(gc)), 'gc_q25': float(np.quantile(gc, .25)), 'gc_q75': float(np.quantile(gc, .75)),
            'cds_length_median': float(np.median(lengths)),
            'mean_pairwise_cosine': float(np.clip(peer_cosines.mean(), -1, 1)) if n > 1 else None,
            'boundary_weight_fraction': float(1-sum(strengths.values())/total_strength) if total_strength else None,
            'top_genes': ';'.join(r['gene_symbol'] for r in cluster_rows[:top_n])})
    return records, summaries, representatives


def eta_squared(values, labels):
    values = np.asarray(values, dtype=float)
    total = np.sum((values-values.mean())**2)
    return float(sum(np.sum(labels == c)*(values[labels == c].mean()-values.mean())**2
                     for c in np.unique(labels))/total) if total else 0.


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def coordinates_for(database, genes, vectors, output, coordinate_path, seed):
    from .reference_umap import load_reference
    coordinate_path = Path(coordinate_path) if coordinate_path else Path(database).parent/'plots/gene_umap_coordinates.csv'
    if not coordinate_path.exists():
        raise ValueError('Generate the saved reference-only UMAP first with llm_mapper.gene_umap')
    manifest_path = coordinate_path.parent/'plot_manifest.json'
    if not manifest_path.exists():raise ValueError('Reused UMAP requires plot_manifest.json provenance')
    provenance=json.loads(manifest_path.read_text());matrix_hash=file_hash(Path(database)/'gene_vectors.npy')
    if provenance['gene_vectors_sha256']!=matrix_hash:raise ValueError('UMAP matrix fingerprint mismatch')
    model=load_reference(coordinate_path.parent,matrix_hash,[g['gene_symbol'] for g in genes])
    with coordinate_path.open() as handle:rows=list(csv.DictReader(handle))
    by_gene={r['gene_symbol']:r for r in rows}
    if len(by_gene)!=len(rows) or set(by_gene)!={g['gene_symbol'] for g in genes}:raise ValueError('UMAP gene labels differ from database')
    xy=np.array([[float(by_gene[g['gene_symbol']][c]) for c in ['UMAP1','UMAP2']] for g in genes])
    if not np.array_equal(xy,model.embedding_):raise ValueError('Coordinates differ from saved reference-only fit')
    info=dict(mode='saved_reference_only_fit',coordinates_sha256=file_hash(coordinate_path),
              reference_umap_model_sha256=file_hash(coordinate_path.parent/'reference_umap.joblib'),parameters=provenance['parameters'])
    write_csv(output/'umap_coordinates.csv',[{'gene_symbol':g['gene_symbol'],'UMAP1':float(p[0]),'UMAP2':float(p[1])} for g,p in zip(genes,xy)])
    return xy,info


def plots(output, xy, genes, labels, summaries, representatives):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    count = len(summaries)
    colors = [plt.get_cmap('tab20')(i % 20) if count <= 20 else plt.get_cmap('turbo')(i/max(count-1,1)) for i in range(count)]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout='constrained')
    axes[0].scatter(*xy.T, c=[colors[i] for i in labels], s=4, linewidths=0, rasterized=True)
    for c in range(count):
        center = np.median(xy[labels == c], axis=0)
        axes[0].text(*center, str(c), ha='center', weight='bold', fontsize=10,
                     bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1.5))
    axes[0].set_title(f'Louvain clusters · {count} communities')
    for ax, values, title in [(axes[1], [g['gc_fraction'] for g in genes], 'GC fraction'),
                              (axes[2], np.log10([g['sequence_length'] for g in genes]), 'CDS length (log10 bases)')]:
        artist=ax.scatter(*xy.T,c=values,s=4,linewidths=0,cmap='viridis',rasterized=True)
        fig.colorbar(artist,ax=ax,shrink=.75);ax.set_title(title)
    for ax in axes: ax.set(xticks=[],yticks=[],xlabel='UMAP 1',ylabel='UMAP 2')
    fig.suptitle('Gene embedding communities and sequence composition · same UMAP projection',fontsize=16)
    for ext in ['png','pdf']: fig.savefig(output/f'cluster_umap.{ext}',dpi=220)
    plt.close(fig)
    fig, axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    axes[0,0].bar(range(count),[s['genes'] for s in summaries],color=colors);axes[0,0].set_ylabel('Genes')
    axes[0,1].bar(range(count),[s['mean_pairwise_cosine'] or 0 for s in summaries],color=colors)
    axes[0,1].set_ylabel('Mean within-cluster cosine');axes[0,1].set_ylim(0,1)
    for ax, values, ylabel in [(axes[1,0],np.array([g['gc_fraction'] for g in genes]),'GC fraction'),
                               (axes[1,1],np.log10([g['sequence_length'] for g in genes]),'CDS length (log10 bases)')]:
        box=ax.boxplot([values[labels==c] for c in range(count)],positions=range(count),patch_artist=True,showfliers=False)
        for patch,color in zip(box['boxes'],colors):patch.set_facecolor(color)
        ax.set_ylabel(ylabel)
    for ax in axes.flat:ax.set_xticks(range(count));ax.set_xlabel('Cluster (ordered by size)')
    fig.suptitle('Cluster QC · composition and cohesion',fontsize=16)
    for ext in ['png','pdf']:fig.savefig(output/f'cluster_qc.{ext}',dpi=200)
    plt.close(fig)
    with PdfPages(output/'top_genes_all_clusters.pdf') as pdf:
        for summary in summaries:
            c=summary['cluster'];rows=[r for r in representatives if r['cluster']==c][::-1]
            fig,ax=plt.subplots(figsize=(9,8),layout='constrained')
            ax.barh([r['gene_symbol'] for r in rows],[r['cosine_degree_centrality'] for r in rows],color=colors[c])
            ax.set_xlabel('Within-cluster cosine-weighted degree / (cluster size − 1)')
            ax.set_title(f'Cluster {c} · {summary["genes"]:,} genes\nTop {len(rows)} genes by subnetwork cosine centrality',loc='left',pad=18)
            ax.text(0,-.10,f'Median GC {summary["gc_median"]:.3f}  ·  median CDS {summary["cds_length_median"]:,.0f} bp',transform=ax.transAxes,color='#555555')
            fig.savefig(output/f'cluster_{c:02d}_top_genes.png',dpi=180);pdf.savefig(fig);plt.close(fig)


def analyze(database, output, neighbors=30, resolution=.3, seed=42, top_n=20, coordinates=None, block_size=256):
    started=time.perf_counter();output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose a new, empty analysis output directory')
    if top_n<1:raise ValueError('top_n must be positive')
    audit=verify_database(database)
    db=GeneVectorDatabase(database)
    vectors=np.load(Path(database)/'gene_vectors.npy',mmap_mode='r',allow_pickle=False)
    output.mkdir(parents=True,exist_ok=True)
    print(f'Building exact cosine {neighbors}-nearest-neighbor graph for {len(vectors):,} genes',flush=True)
    graph,adjacency=cosine_graph(vectors,neighbors,block_size)
    sparse.save_npz(output/'cosine_knn_graph.npz',adjacency)
    print('Running Louvain and second-seed stability check',flush=True)
    communities,labels=partition(graph,resolution,seed)
    alternate,other_labels=partition(graph,resolution,seed+1)
    records,summaries,representatives=characterize(graph,vectors,db.genes,communities,labels,top_n)
    write_csv(output/'gene_clusters.csv',records);write_csv(output/'cluster_summary.csv',summaries)
    write_csv(output/'top_genes_per_cluster.csv',representatives)
    write_csv(output/'graph_nodes.csv',[{'index':i,'gene_symbol':g['gene_symbol'],'cluster':int(labels[i])} for i,g in enumerate(db.genes)])
    sample_size=min(2000,len(vectors))
    rng=np.random.default_rng(seed);sample=rng.choice(len(vectors),sample_size,replace=False)
    silhouette=None
    if 1<len(np.unique(labels[sample]))<sample_size:
        silhouette=float(silhouette_score(vectors[sample],labels[sample],metric='cosine'))
    xy,projection=coordinates_for(database,db.genes,vectors,output,coordinates,seed)
    gc=np.array([g['gc_fraction'] for g in db.genes]);loglength=np.log10([g['sequence_length'] for g in db.genes])
    qc={'genes':len(vectors),'clusters':len(communities),'edges':graph.number_of_edges(),
        'connected_components':nx.number_connected_components(graph),'isolates':len(list(nx.isolates(graph))),
        'singleton_clusters':sum(len(c)==1 for c in communities),
        'modularity_at_resolution':nx.community.modularity(graph,communities,weight='weight',resolution=resolution) if graph.number_of_edges() else None,
        'silhouette_cosine':silhouette,'silhouette_sample_size':sample_size,'silhouette_seed':seed,
        'second_seed':seed+1,'second_seed_clusters':len(alternate),'second_seed_adjusted_rand_index':float(adjusted_rand_score(labels,other_labels)),
        'gc_eta_squared':eta_squared(gc,labels),'log10_cds_length_eta_squared':eta_squared(loglength,labels),
        'gc_umap_spearman':[float(spearmanr(gc,xy[:,axis]).statistic) if np.ptp(gc)>0 and np.ptp(xy[:,axis])>0 else None for axis in range(2)],
        'database_integrity':audit['status']}
    write_json(output/'qc.json',qc)
    print(f'Plotting {len(communities)} clusters and representative genes',flush=True)
    plots(output,xy,db.genes,labels,summaries,representatives)
    manifest={'status':'complete','parameters':{'neighbors':neighbors,'resolution':resolution,'seed':seed,'top_n':top_n,'block_size':block_size},
        'graph':'Exact cosine kNN, union symmetrized with max weight; positive edges only; no self loops; ties by database row index',
        'centrality':'Sum of internal edge cosine weights divided by cluster size minus one; rank ties by mean cosine to all cluster peers, then symbol',
        'cluster_ids':'Zero-based, descending cluster size, ties by smallest database row index',
        'gene_vectors_sha256':file_hash(Path(database)/'gene_vectors.npy'),'database_signature':db.manifest['signature'],
        'projection':projection,'seconds':time.perf_counter()-started,
        'packages':{p:version(p) for p in ['numpy','scipy','networkx','scikit-learn','matplotlib']}}
    report=['# Gene embedding Louvain analysis','',f"{qc['genes']:,} genes; {qc['clusters']} clusters; resolution {resolution}; cosine kNN k={neighbors}.",'',
        f"GC eta-squared: {qc['gc_eta_squared']:.3f}; log10 CDS length eta-squared: {qc['log10_cds_length_eta_squared']:.3f}.",
        f"Second-seed adjusted Rand index: {qc['second_seed_adjusted_rand_index']:.3f}; sampled cosine silhouette: {silhouette}.",'',
        'Cluster IDs are ordered by size. Centrality ranks graph hubs within the cluster, not expression or biological importance.',
        'GC/length eta-squared is the fraction of feature variance explained by cluster membership, not a causal estimate.',
        'UMAP coordinates are used only for display. Axis orientation is arbitrary. The graph uses original embeddings.',
        'Composition and high background cosine similarity can drive clusters; representatives are hypotheses for biological review, not enrichment results.',
        'Resolution and neighbor count affect communities; a second seed is a limited stability check, not a parameter robustness study.','',
        '| Cluster | Genes | Median GC | Median CDS bp | Top 5 central genes |','|---|---:|---:|---:|---|']
    for s in summaries:report.append(f"| {s['cluster']} | {s['genes']} | {s['gc_median']:.3f} | {s['cds_length_median']:.0f} | {', '.join(s['top_genes'].split(';')[:5])} |")
    (output/'report.md').write_text('\n'.join(report)+'\n')
    write_json(output/'analysis_manifest.json',manifest)
    print(json.dumps(qc,indent=2),flush=True)
    return qc


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database');parser.add_argument('output')
    parser.add_argument('--neighbors',type=int,default=30);parser.add_argument('--resolution',type=float,default=.3)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--top-n',type=int,default=20)
    parser.add_argument('--coordinates',help='Saved reference-only gene_umap_coordinates.csv; defaults to the database sibling plots/ directory')
    parser.add_argument('--block-size',type=int,default=256)
    args=parser.parse_args(argv)
    analyze(args.database,args.output,args.neighbors,args.resolution,args.seed,args.top_n,args.coordinates,args.block_size)


if __name__=='__main__':main()
