import numpy as np
import pytest
from scripts.characterize_gene_clusters import enrichment


def test_fisher_tail_and_log_tail_agree():
    result=enrichment(np.array([1,1,0,0],dtype=bool),np.array([1,0,0,0],dtype=bool))
    assert result['cluster_positive']==1 and result['rest_positive']==0
    assert result['p']==pytest.approx(.5)
    assert 10**result['log10_p']==pytest.approx(.5)


def test_extreme_enrichment_preserves_finite_log_probability():
    cluster=np.arange(20000)<500
    feature=np.arange(20000)<450
    result=enrichment(cluster,feature)
    assert result['p']==0
    assert np.isfinite(result['log10_p']) and result['log10_p'] < -300
    assert result['odds_ratio'] is None and result['odds_ratio_nonfinite']=='inf'
