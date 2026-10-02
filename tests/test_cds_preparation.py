import csv,json
from pathlib import Path
import pysam
from llm_mapper.preparation.cds import extract_cds
from llm_mapper.extract_gene_sequences import reverse_complement


def test_transcript_selection_strand_and_phase(tmp_path):
    seq='AACCGGTTACGTACGTTTTTCCCCAAAAGGGG'
    fa=tmp_path/'ref.fa';fa.write_text('>1\n'+seq+'\n');pysam.faidx(str(fa))
    genes=tmp_path/'genes.csv'
    with genes.open('w') as h:
        w=csv.DictWriter(h,fieldnames=['gene_symbol','gene_id','chrom','strand','start','end'])
        w.writeheader()
        w.writerows([dict(gene_symbol='PLUS',gene_id='G1',chrom='1',strand='+',start=1,end=30),
                     dict(gene_symbol='MINUS',gene_id='G2',chrom='1',strand='-',start=1,end=30),
                     dict(gene_symbol='MISSING',gene_id='G3',chrom='1',strand='+',start=1,end=30)])
    gtf=tmp_path/'genes.gtf';lines=[]
    def row(feature,start,end,strand,gene,tx,tags='',phase='0'):
        lines.append(f'1\ttest\t{feature}\t{start}\t{end}\t.\t{strand}\t{phase}\tgene_id "{gene}"; transcript_id "{tx}"; {tags}\n')
    row('transcript',1,30,'+','G1','MANE','tag "MANE_Select"; tag "basic";')
    row('CDS',1,3,'+','G1','MANE');row('CDS',8,11,'+','G1','MANE',phase='1')
    row('transcript',1,30,'+','G1','LONG','tag "Ensembl_canonical";')
    row('CDS',1,25,'+','G1','LONG')
    row('transcript',1,30,'-','G2','MINUS_TX','tag "Ensembl_canonical"; tag "basic";')
    row('CDS',2,5,'-','G2','MINUS_TX');row('CDS',15,20,'-','G2','MINUS_TX')
    gtf.write_text(''.join(lines))
    out=tmp_path/'cds';r=extract_cds(fa,gtf,genes,out)
    assert r['genes_written']==2 and len(r['unresolved'])==1
    with (out/'genes.csv').open() as h:data={r['gene_symbol']:r for r in csv.DictReader(h)}
    assert data['PLUS']['transcript_id']=='MANE'
    assert data['PLUS']['sequence']==seq[0:3]+seq[7:11]
    assert data['MINUS']['sequence']==reverse_complement(seq[14:20])+reverse_complement(seq[1:5])
    assert data['MINUS']['sequence']==reverse_complement(seq[1:5]+seq[14:20])
    blocks=json.loads(data['MINUS']['cds_parts'])
    assert blocks[0]['genomic_start']==14 and blocks[0]['cds_start']==0
