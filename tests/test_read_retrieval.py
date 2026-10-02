import json
import numpy as np
import pytest
from llm_mapper.read_retrieval import IntervalLabels,classification_metrics,cosine_retrieve,select_reads


def test_overlap_truth_is_half_open_and_detects_nested_genes():
    rows=[dict(chrom='1',genomic_start=a,genomic_end=b,gene_symbol=g) for a,b,g in [(10,100,'A'),(20,30,'B'),(101,120,'C')]]
    index=IntervalLabels(rows)
    assert index.overlaps('1',30,40)=={'A'}
    assert index.overlaps('1',25,35)=={'A','B'}
    assert index.overlaps('1',100,101)==set()


def test_specificity_counts_other_hgnc_predictions_as_false_negatives():
    result=classification_metrics(['A','A','B','B'],['A','X','A','B'],['A','B','C'])
    assert result['accuracy']==.5
    a,b,c=result['per_gene']
    assert (a['tp'],a['fn'],a['fp'],a['tn'])==(1,1,1,1)
    assert a['sensitivity']==.5 and a['specificity']==.5
    assert b['specificity']==1 and c['sensitivity'] is None


def test_cosine_retrieval_uses_both_strands_without_truth_and_stable_ties():
    refs=np.array([[1.,0],[0,1],[1,0]])
    forward=np.array([[0.,1],[.6,.8]])
    reverse=np.array([[1.,0],[.8,.6]])
    result=cosine_retrieve(forward,reverse,refs,[0,1,2])
    assert result['predicted'].tolist()==[0,0]
    assert result['orientation'].tolist()==['reverse_complement','reverse_complement']
    restricted=cosine_retrieve(forward,reverse,refs,[1])
    assert restricted['predicted'].tolist()==[1,1]


def test_highest_as_reservoir_filters_direct_cram_and_reproduces(tmp_path):
    pysam=pytest.importorskip('pysam')
    fasta=tmp_path/'reference.fa';fasta.write_text('>chr1\n'+'A'*1000+'\n');pysam.faidx(str(fasta))
    cram=tmp_path/'reads.cram'
    header=dict(HD={'VN':'1.6','SO':'coordinate'},SQ=[dict(SN='chr1',LN=1000)],RG=[dict(ID='rg',SM='sample')])
    with pysam.AlignmentFile(str(cram),'wc',header=header,reference_filename=str(fasta)) as out:
        for i,(score,mapq,flag) in enumerate([(90,60,0),(100,60,0),(100,60,0),(110,255,0),(110,60,1024)]):
            read=pysam.AlignedSegment();read.query_name=f'r{i}';read.query_sequence='A'*100;read.flag=flag
            read.reference_id=0;read.reference_start=110+i;read.mapping_quality=mapq;read.cigarstring='100M'
            read.query_qualities=pysam.qualitystring_to_array('I'*100);read.set_tag('AS',score);read.set_tag('RG','rg');out.write(read)
    pysam.index(str(cram))
    blocks=tmp_path/'blocks.jsonl';blocks.write_text(json.dumps(dict(chrom='1',genomic_start=100,genomic_end=500,gene_symbol='A',strand='+',transcript_id='tx'))+'\n')
    result=select_reads(cram,fasta,blocks,tmp_path/'selected',targets=['A'],per_gene=1,seed=7)
    assert len(result)==1 and result[0]['alignment_score']==100
    report=json.loads((tmp_path/'selected/selection_manifest.json').read_text())
    assert report['genes'][0]['eligible_reads']==3 and report['genes'][0]['highest_AS_tier_reads']==2
    second=select_reads(cram,fasta,blocks,tmp_path/'second',targets=['A'],per_gene=1,seed=7)
    assert second==result
    assert select_reads(cram,fasta,blocks,tmp_path/'selected',targets=['A'],per_gene=1,seed=7)==result
