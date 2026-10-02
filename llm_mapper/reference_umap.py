"""Persist a reference-only UMAP fit for immutable out-of-sample projection."""
from importlib.metadata import version
import json
from pathlib import Path
import joblib
import numpy as np
from .preparation.downloads import file_hash, write_json


def fit_reference(vectors, output, parameters, matrix_hash, symbols):
    import umap
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    signature=dict(parameters=parameters,gene_vectors_sha256=matrix_hash,gene_symbols=list(symbols),
                   packages={p:version(p) for p in ['umap-learn','numpy','scikit-learn','numba','joblib']})
    manifest=output/'reference_umap_manifest.json'
    if manifest.exists():
        prior=json.loads(manifest.read_text())
        if prior['signature']!=signature:raise ValueError('Reference UMAP signature mismatch')
        model=load_reference(output,matrix_hash,symbols)
    else:
        model=umap.UMAP(**parameters).fit(np.asarray(vectors))
        path=output/'reference_umap.joblib';temporary=path.with_suffix('.tmp')
        joblib.dump(model,temporary,compress=3);temporary.replace(path)
        write_json(manifest,dict(status='complete',fit_scope='reference gene vectors only; no query reads',
                   signature=signature,model_sha256=file_hash(path),trusted_local_pickle_only=True))
    xy=np.asarray(model.embedding_)
    if xy.shape!=(len(vectors),2) or not np.isfinite(xy).all():raise ValueError('Invalid reference coordinates')
    return model,xy.copy()


def load_reference(output,matrix_hash,symbols):
    """Load only a trusted local artifact; never accept untrusted pickle files."""
    output=Path(output);m=json.loads((output/'reference_umap_manifest.json').read_text())
    if m['status']!='complete' or m['signature']['gene_vectors_sha256']!=matrix_hash or m['signature']['gene_symbols']!=list(symbols):
        raise ValueError('Reference UMAP does not match database/order')
    if any(version(p)!=v for p,v in m['signature']['packages'].items()):raise ValueError('UMAP environment versions changed')
    if file_hash(output/'reference_umap.joblib')!=m['model_sha256']:raise ValueError('UMAP model checksum mismatch')
    model=joblib.load(output/'reference_umap.joblib')
    return model


def project_queries(model,vectors):
    before=model.embedding_.copy()
    xy=model.transform(np.asarray(vectors,dtype=np.float32))
    if not np.array_equal(before,model.embedding_):raise AssertionError('Query transform changed reference embedding')
    if xy.shape!=(len(vectors),2) or not np.isfinite(xy).all():raise ValueError('Invalid query coordinates')
    return xy
