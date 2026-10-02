"""Regenerate existing study views after recovering an identical reference-only UMAP fit.

Refuses coordinate changes: a changed fit needs a separate provenance migration,
not silent replacement of coordinates attached to completed analyses.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from llm_mapper.gene_database import GeneVectorDatabase
from llm_mapper.gene_clusters import plots
from llm_mapper.neighborhood_alignment import draw
from llm_mapper.alignment_permutations import summarize
from llm_mapper.reference_umap import load_reference
from llm_mapper.preparation.downloads import file_hash,write_json


def rows(path):
    with path.open() as h:return list(csv.DictReader(h))


def typed(path):
    def value(x):
        try:return json.loads(x)
        except (ValueError,TypeError):return x
    return [{k:value(v) for k,v in row.items()} for row in rows(path)]


def refresh(root):
    root=Path(root);db=GeneVectorDatabase(root/'vector_db');symbols=[g['gene_symbol'] for g in db.genes]
    model=load_reference(root/'plots',file_hash(root/'vector_db/gene_vectors.npy'),symbols);xy=model.embedding_
    cluster=root/'louvain_r0.3_k30';sw=root/'sw_neighborhood';null=root/'sw_stratified_null'
    for path in [root/'plots/gene_umap_coordinates.csv',cluster/'umap_coordinates.csv',sw/'gene_neighbor_sw.csv']:
        records=rows(path);by_symbol={r['gene_symbol']:r for r in records}
        old=np.array([[float(by_symbol[s][k]) for k in ['UMAP1','UMAP2']] for s in symbols])
        if not np.array_equal(old,xy):raise ValueError(f'Changed reference coordinates: explicit migration required for {path.name}')
    nodes=rows(cluster/'graph_nodes.csv');labels=np.array([int(r['cluster']) for r in nodes])
    scores=rows(sw/'gene_neighbor_sw.csv');observed=np.array([float(r['mean_sw_similarity']) for r in scores])
    before={str(p.relative_to(root)):file_hash(p) for p in [cluster/'cosine_knn_graph.npz',cluster/'gene_clusters.csv',sw/'gene_neighbor_sw.csv',null/'gene_sw_null.csv',null/'summary.json']}
    plots(cluster,xy,db.genes,labels,typed(cluster/'cluster_summary.csv'),typed(cluster/'top_genes_per_cluster.csv'))
    draw(sw,xy,observed,db.genes,labels)
    manifest=json.loads((null/'manifest.json').read_text());values=[]
    for checkpoint in manifest['completed']:
        path=null/checkpoint['file']
        if file_hash(path)!=checkpoint['sha256']:raise ValueError('Null checkpoint changed')
        values.append(np.load(path,allow_pickle=False))
    sizes=np.array([int(r['stratum_size']) for r in rows(null/'strata.csv')])
    summarize(null,observed,np.array(values),nodes,xy,sizes)
    for relative,digest in before.items():
        if file_hash(root/relative)!=digest:raise AssertionError(f'Nonvisual analysis changed: {relative}')
    umaps=sorted(p for p in root.rglob('*') if p.suffix in ['.png','.pdf','.svg'] and 'umap' in p.name)
    report=dict(status='complete',fit_scope='reference-only',reference_model_sha256=file_hash(root/'plots/reference_umap.joblib'),
                coordinate_change_max_absolute=0.0,reference_coordinates_identical_to_previous=True,
                graph_scores_and_statistics_unchanged=True,validated_statistic_hashes=before,
                regenerated_umap_artifacts={str(p.relative_to(root)):file_hash(p) for p in umaps})
    write_json(root/'reference_umap_refresh.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');print(json.dumps(refresh(p.parse_args().root),indent=2))
