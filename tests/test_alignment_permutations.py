import numpy as np
import pytest
from llm_mapper.alignment_permutations import make_strata, permutation, gene_means, null_statistics
from llm_mapper.alignment import SmithWaterman
from llm_mapper.alignment.smith_waterman import reverse_complement


def test_permutation_preserves_strata_and_degree_and_is_reproducible():
    gc=np.tile([.3,.4,.5,.6],20);lengths=np.tile([100,200,300,400],20)
    degree=np.repeat([3,4],40)
    groups,labels,sizes,bounds=make_strata(gc,lengths,degree,2)
    p=permutation(groups,len(gc),123,0)
    assert np.array_equal(np.sort(p),np.arange(len(gc)))
    assert np.array_equal(labels[p],labels)
    assert np.array_equal(degree[p],degree)
    assert np.array_equal(p,permutation(groups,len(gc),123,0))
    assert not np.array_equal(p,permutation(groups,len(gc),123,1))
    assert np.all(sizes>=10)


def test_gene_following_means_keep_zero_edges_and_original_gene_order():
    # Triangle, assignment[node]=gene. Node sums are 0.6,0.2,0.8.
    values=gene_means(np.array([0,0,1]),np.array([1,2,2]),np.array([0,.6,.2]),
                      np.array([2,2,2]),np.array([2,0,1]))
    np.testing.assert_allclose(values,[.1,.4,.3])


def test_empirical_p_ties_and_bh():
    null=np.array([[.2,.2,.2],[.3,.3,.3],[.4,.4,.4]])
    stats=null_statistics(np.array([.5,.3,.1]),null)
    np.testing.assert_allclose(stats['p_enrichment'],[.25,.75,1])
    np.testing.assert_allclose(stats['q_bh_exploratory'],[.75,1,1])
    np.testing.assert_allclose(stats['excess_similarity'],[.2,0,-.2],atol=1e-15)


def test_precomputed_reverse_complement_scoring_matches_observed_engine():
    pytest.importorskip('parasail');engine=SmithWaterman()
    sequences=['ACGTACGT','AAGTC','GACTT','ACNT','TTTTACGTACGT']
    for a in sequences:
        for b in sequences:
            score=max(engine.raw_score(a,b),engine.raw_score(a,reverse_complement(b)))
            sim=2*score/(2*(len(a)+len(b)-a.count('N')-b.count('N')))
            assert sim==pytest.approx(engine.compare(a,b)['similarity'])


def test_checkpoint_resume_and_provenance(tmp_path,monkeypatch):
    import csv,json
    from types import SimpleNamespace
    from dataclasses import asdict
    from scipy import sparse
    from llm_mapper.alignment_permutations import run
    from llm_mapper.preparation.downloads import file_hash
    engine=SmithWaterman();sequences=['ACGT','ACGA','ACTT','TTTT']
    nodes=[dict(gene_symbol=str(i),cluster=0) for i in range(4)]
    db=SimpleNamespace(genes=[dict(gc_fraction=.5) for _ in nodes])
    graph=sparse.csr_matrix(np.ones((4,4))-np.eye(4))
    signature=dict(test_fingerprint='fixed')
    monkeypatch.setattr('llm_mapper.alignment_permutations.load_inputs',lambda *a:(db,graph,sequences,nodes,np.zeros((4,2)),signature.copy()))
    monkeypatch.setattr('llm_mapper.alignment_permutations.summarize',lambda *a:None)
    observed=tmp_path/'observed';observed.mkdir();out=tmp_path/'null'
    sig=dict(signature,scoring=asdict(engine.config),parasail_library_sha256=file_hash(engine.parasail.get_library()))
    (observed/'manifest.json').write_text(json.dumps(dict(status='complete',signature=sig)))
    actual=np.array([np.mean([engine.compare(a,b)['similarity'] for j,b in enumerate(sequences) if j!=i]) for i,a in enumerate(sequences)])
    with (observed/'gene_neighbor_sw.csv').open('w') as h:
        writer=csv.writer(h);writer.writerow(['gene_symbol','mean_sw_similarity'])
        writer.writerows(zip(map(str,range(4)),actual))
    args=('db','graph','fasta',observed,out)
    first=run(*args,permutations=3,pilot=1,bins=1,workers=2,stop_after=1)
    assert len(first['completed'])==1
    final=run(*args,permutations=3,pilot=1,bins=1,workers=2)
    assert final['status']=='complete' and len(final['completed'])==3
    assert final['completed'][0]==first['completed'][0]
    # Complete graph keeps every gene's neighbors unchanged under any labeling.
    for record in final['completed']:
        np.testing.assert_allclose(np.load(out/record['file']),actual)
    with pytest.raises(ValueError,match='signature mismatch'):
        run(*args,permutations=3,pilot=1,bins=1,seed=7)
    path=out/final['completed'][0]['file'];path.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):
        run(*args,permutations=3,pilot=1,bins=1)
