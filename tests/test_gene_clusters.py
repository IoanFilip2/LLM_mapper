import networkx as nx
import numpy as np
import pytest
from llm_mapper.gene_clusters import cosine_graph, partition, characterize, eta_squared


def test_graph_removes_self_loops_even_with_duplicate_vectors():
    values=np.array([[1.,0.],[1.,0.],[.8,.6],[0.,1.]],dtype=np.float32)
    graph,adjacency=cosine_graph(values,neighbors=1,block_size=2)
    assert list(nx.selfloop_edges(graph))==[]
    assert graph[0][1]['weight']==pytest.approx(1.)
    assert (adjacency-adjacency.T).nnz==0
    assert all(d['weight']>0 for _,_,d in graph.edges(data=True))
    with pytest.raises(ValueError):cosine_graph(values,neighbors=4)


def test_partition_and_graph_strength_ranking():
    graph=nx.Graph();graph.add_nodes_from(range(4))
    graph.add_weighted_edges_from([(0,1,.9),(0,2,.8),(1,2,.1)])
    communities,labels=partition(graph,.3,42)
    assert communities==[{0,1,2},{3}]
    values=np.eye(4,dtype=np.float32)
    genes=[dict(gene_symbol=f'G{i}',gc_fraction=.4+i*.1,sequence_length=100+i) for i in range(4)]
    records,summary,top=characterize(graph,values,genes,communities,labels,top_n=2)
    assert top[0]['gene_symbol']=='G0'
    assert top[0]['cosine_degree_centrality']==pytest.approx(.85)
    assert top[0]['mean_cosine_to_cluster_peers']==pytest.approx(0.)
    assert summary[1]['mean_pairwise_cosine'] is None
    assert len(top)==3
    assert eta_squared([0,0,1,1],np.array([0,0,1,1]))==pytest.approx(1.)


def test_no_positive_edges_and_invalid_resolution():
    graph,_=cosine_graph(np.eye(3,dtype=np.float32),neighbors=1)
    communities,labels=partition(graph)
    assert len(communities)==3
    with pytest.raises(ValueError):partition(graph,resolution=0)


def test_clustering_does_not_silently_fit_another_umap(tmp_path):
    from llm_mapper.gene_clusters import coordinates_for
    with pytest.raises(ValueError,match='saved reference-only UMAP'):
        coordinates_for(tmp_path/'db',[],np.empty((0,2)),tmp_path,None,42)
