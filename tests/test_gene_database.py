import csv
from dataclasses import replace
import json
import numpy as np
import pytest
from llm_mapper.embedders import SequenceEmbedder,EmbeddingConfig
from llm_mapper.gene_database import build_gene_database,GeneVectorDatabase
from llm_mapper.index import normalize,VectorIndex
from llm_mapper.verify_gene_database import verify_database


class TinyEmbedder(SequenceEmbedder):
    config=EmbeddingConfig(model='test-model')
    device='cpu'
    dimension=4
    def encode(self,sequences,batch_size=8):
        return np.array([[s.count(b)/len(s) for b in 'ACGT'] for s in sequences],dtype=np.float32)


def inputs(tmp_path):
    fa=tmp_path/'genes.fa';meta=tmp_path/'genes.csv'
    seqs={'LONG':'AAAACCGGTT','SHORT':'ACGT','MEDIUM':'CCCCCC'}
    fa.write_text(''.join(f'>{name}\n{seq}\n' for name,seq in seqs.items()))
    with meta.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['gene_symbol','sequence_length','sequence'])
        writer.writeheader()
        writer.writerows({'gene_symbol':n,'sequence_length':len(s),'sequence':s} for n,s in seqs.items())
    return fa,meta,seqs


def test_coverage_weighting_resume_and_search(tmp_path):
    fasta,meta,seqs=inputs(tmp_path);out=tmp_path/'db';model=TinyEmbedder()
    result=build_gene_database(fasta,meta,model,out,chunk_length=4,batch_size=2,genes_per_checkpoint=1,max_blocks=1)
    assert result['status']=='building' and result['completed_genes']==1
    with pytest.raises(ValueError,match='incomplete'):GeneVectorDatabase(out)
    result=build_gene_database(fasta,meta,model,out,chunk_length=4,batch_size=2,genes_per_checkpoint=1)
    assert result['status']=='complete' and result['completed_windows']==6
    db=GeneVectorDatabase(out,model)
    values=np.load(out/'gene_vectors.npy')
    expected=normalize(model.encode([seqs[r['gene_symbol']] for r in db.genes]))
    np.testing.assert_allclose(values,expected,atol=1e-6)
    assert db.search(['CCCC'],k=1)[0][0]['gene_symbol']=='MEDIUM'
    hit=db.search(['AAAA'],k=1,level='window')[0][0]
    assert hit['gene_symbol']=='LONG' and hit['gene_sequence_start']==0 and hit['gene_sequence_end']==4
    assert VectorIndex(out/'gene_index',model).search(['CCCC'])[0]['label']=='MEDIUM'
    audit=verify_database(out)
    assert audit['status']=='passed' and audit['coding_bases']==20
    assert audit['checkpoint_blocks_verified']==3
    assert len(result['execution_history'])==2
    with pytest.raises(ValueError,match='mismatch'):
        build_gene_database(fasta,meta,model,out,chunk_length=5,genes_per_checkpoint=1)


def test_corrupted_checkpoint_is_rejected(tmp_path):
    fasta,meta,_=inputs(tmp_path);out=tmp_path/'db';model=TinyEmbedder()
    build_gene_database(fasta,meta,model,out,chunk_length=4,genes_per_checkpoint=1,max_blocks=1)
    vectors=np.load(out/'window_vectors.npy',mmap_mode='r+')
    vectors[0,0]=123.;vectors.flush()
    with pytest.raises(ValueError,match='checksum'):
        build_gene_database(fasta,meta,model,out,chunk_length=4,genes_per_checkpoint=1)


def test_dimension_inferred_for_generic_embedder(tmp_path):
    class DynamicEmbedder(TinyEmbedder):
        dimension=None
    fasta,meta,_=inputs(tmp_path)
    model=DynamicEmbedder()
    result=build_gene_database(fasta,meta,model,tmp_path/'db',chunk_length=4,genes_per_checkpoint=1)
    assert result['signature']['dimension']==4


def test_completed_audit_rejects_normalized_but_corrupted_vectors(tmp_path):
    fasta,meta,_=inputs(tmp_path);out=tmp_path/'db'
    build_gene_database(fasta,meta,TinyEmbedder(),out,chunk_length=4,genes_per_checkpoint=1)
    vectors=np.load(out/'gene_vectors.npy',mmap_mode='r+')
    vectors[0]=[1.,0.,0.,0.];vectors.flush()
    with pytest.raises(ValueError,match='checksum'):
        verify_database(out)
