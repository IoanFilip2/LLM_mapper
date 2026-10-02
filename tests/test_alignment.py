import numpy as np
import pytest
from scipy import sparse
from llm_mapper.alignment import SWConfig, SmithWaterman, reference_sw_score
from llm_mapper.alignment.smith_waterman import reverse_complement
from llm_mapper.neighborhood_alignment import aggregate_neighbors


@pytest.mark.parametrize('algorithm',['striped','scan'])
def test_simd_matches_independent_affine_dp(algorithm):
    pytest.importorskip('parasail')
    c=SWConfig(algorithm=algorithm);e=SmithWaterman(c);rng=np.random.default_rng(7)
    for _ in range(30):
        a='A'+''.join(rng.choice(list('ACGTN'),size=int(rng.integers(1,45))))
        b='C'+''.join(rng.choice(list('ACGTN'),size=int(rng.integers(1,45))))
        assert e.raw_score(a,b)==reference_sw_score(a,b,c)
    assert e.raw_score('A'*10+'G'*3+'C'*10,'A'*10+'C'*10)==33


def test_similarity_normalization_strand_and_length():
    pytest.importorskip('parasail');e=SmithWaterman()
    assert e.compare('AAGTC',reverse_complement('AAGTC'))['similarity']==1
    assert e.compare('AAGTC',reverse_complement('AAGTC'))['orientation']=='reverse_complement'
    assert e.compare('ACGT','ACGTACGT')['similarity']==pytest.approx(2/3)
    assert e.compare('ACGT','ACGTACGT')['shorter_normalized_similarity']==1
    assert e.compare('ACNT','ACNT')['similarity']==1
    assert e.compare('ACGT','ACGTTT')==e.compare('ACGTTT','ACGT')
    for bad in ['', 'NNNN', 'ACGU']:
        with pytest.raises(ValueError):e.compare(bad,'ACGT')


def test_long_score_avoids_16bit_overflow():
    pytest.importorskip('parasail');e=SmithWaterman(SWConfig(both_strands=False))
    assert e.compare('A'*17000,'A'*17000)['score']==34000


def test_sparse_aggregation_keeps_zero_score_neighbors_and_isolates():
    a=sparse.csr_matrix([[0,.8,.2,0],[.8,0,0,0],[.2,0,0,0],[0,0,0,0]])
    sw,degree,mean,weighted,short=aggregate_neighbors(a,np.array([0,0]),np.array([1,2]),np.array([1.,0.]),np.array([1.,0.]))
    assert mean[0]==pytest.approx(.5) and weighted[0]==pytest.approx(.8)
    assert mean[2]==0 and np.isnan(mean[3])
    assert degree.tolist()==[2,1,1,0]
    assert (sw-sw.T).nnz==0


def test_pipeline_checkpoint_resume_and_provenance(tmp_path,monkeypatch):
    pytest.importorskip('parasail')
    import csv,json
    from llm_mapper.embedders import SequenceEmbedder,EmbeddingConfig
    from llm_mapper.gene_database import build_gene_database,GeneVectorDatabase
    from llm_mapper.gene_clusters import write_csv
    from llm_mapper.preparation.downloads import file_hash
    from llm_mapper.neighborhood_alignment import run
    class Tiny(SequenceEmbedder):
        config=EmbeddingConfig(model='test');device='cpu';dimension=4
        def encode(self,sequences,batch_size=2):return np.array([[s.count(b) for b in 'ACGT'] for s in sequences],dtype=np.float32)
    fasta=tmp_path/'genes.fa';meta=tmp_path/'genes.csv';database=tmp_path/'db';graph=tmp_path/'graph';graph.mkdir()
    seqs={'A':'ACGTACGT','B':'ACGTTCGT','C':'ACGTACGTTT','D':'TTTTACGT'}
    fasta.write_text(''.join(f'>{name}\n{seq}\n' for name,seq in seqs.items()))
    write_csv(meta,[{'gene_symbol':n,'sequence_length':len(s),'sequence':s} for n,s in seqs.items()])
    build_gene_database(fasta,meta,Tiny(),database,chunk_length=20)
    db=GeneVectorDatabase(database)
    sparse.save_npz(graph/'cosine_knn_graph.npz',sparse.csr_matrix(np.ones((4,4))-np.eye(4)))
    (graph/'analysis_manifest.json').write_text(json.dumps({'status':'complete','gene_vectors_sha256':file_hash(database/'gene_vectors.npy')}))
    write_csv(graph/'graph_nodes.csv',[{'index':i,'gene_symbol':g['gene_symbol'],'cluster':i%2} for i,g in enumerate(db.genes)])
    write_csv(graph/'umap_coordinates.csv',[{'gene_symbol':g['gene_symbol'],'UMAP1':i,'UMAP2':i%2} for i,g in enumerate(db.genes)])
    monkeypatch.setattr('llm_mapper.neighborhood_alignment.draw',lambda *args:None)
    out=tmp_path/'output'
    first=run(database,graph,fasta,out,workers=2,batch_size=2,max_batches=1)
    assert first['status']=='building' and first['completed_edges']==2
    assert not (out/'gene_neighbor_sw.csv').exists()
    final=run(database,graph,fasta,out,workers=2,batch_size=2)
    assert final['status']=='complete' and final['completed_edges']==6
    rows=list(csv.DictReader((out/'gene_neighbor_sw.csv').open()))
    assert len(rows)==4 and all(int(r['neighbors'])==3 for r in rows)
    assert final['execution_history'][-1]['resume_edges']==2
    with pytest.raises(ValueError,match='signature mismatch'):
        run(database,graph,fasta,out,config=SWConfig(gap_open=6))
