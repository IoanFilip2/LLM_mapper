import numpy as np
import pytest
from llm_mapper.reference_umap import fit_reference,load_reference,project_queries


def test_saved_reference_fit_transforms_queries_without_changing_reference(tmp_path):
    pytest.importorskip('umap')
    rng=np.random.default_rng(123);vectors=rng.normal(size=(25,8)).astype('float32');vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    symbols=[str(i) for i in range(len(vectors))]
    parameters=dict(n_neighbors=4,n_components=2,metric='cosine',random_state=42,n_jobs=1,n_epochs=20)
    model,xy=fit_reference(vectors,tmp_path,parameters,'fingerprint',symbols)
    queries=rng.normal(size=(5,8)).astype('float32');queries/=np.linalg.norm(queries,axis=1,keepdims=True)
    projected=project_queries(model,queries)
    restored=load_reference(tmp_path,'fingerprint',symbols)
    np.testing.assert_array_equal(projected,project_queries(restored,queries))
    np.testing.assert_array_equal(xy,restored.embedding_)
    with pytest.raises(ValueError,match='match database'):
        load_reference(tmp_path,'different',symbols)
