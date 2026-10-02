"""Transcript-aware CDS extraction at the already-reviewed HGNC gene loci."""
import argparse
from collections import defaultdict, Counter
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
from .reference import gtf_records, check_gtf_build
from .downloads import file_hash, write_json
from ..extract_gene_sequences import reverse_complement


def extract_cds(fasta, gtf, genes_csv, output):
    import pysam
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    check_gtf_build(gtf)
    csv.field_size_limit(sys.maxsize)
    with open(genes_csv) as handle:
        genes=[{k:v for k,v in r.items() if k!='sequence'} for r in csv.DictReader(handle)]
    by_id={g['gene_id']:g for g in genes}
    if len(by_id)!=len(genes):
        raise ValueError('Reviewed gene IDs must be unique')
    transcripts=defaultdict(dict)
    for row in gtf_records(gtf,('transcript','CDS')):
        gene=by_id.get(row.get('gene_id','').split('.')[0])
        if not gene or (row['chrom'],row['strand'])!=(gene['chrom'],gene['strand']):
            continue
        if not (int(gene['start'])<=row['start']<=row['end']<=int(gene['end'])):
            raise ValueError('CDS/transcript lies outside selected gene locus')
        tid=row.get('transcript_id')
        if not tid:
            continue
        entry=transcripts[gene['gene_id']].setdefault(tid,{'parts':set(),'tags':set(),'version':row.get('transcript_version','')})
        entry['tags'].update(row['tags'])
        if row['feature']=='CDS':
            entry['parts'].add((row['start'],row['end'],row['phase']))
    def choose(entries):
        candidates=[(tid,e) for tid,e in entries.items() if e['parts']]
        def priority(candidate):
            tid,e=candidate
            rank=0 if 'MANE_Select' in e['tags'] else 1 if 'Ensembl_canonical' in e['tags'] else 2
            return rank,-sum(b-a+1 for a,b,_ in e['parts']),tid
        return min(candidates,key=priority) if candidates else None
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.cds-',dir=output.parent))
    audit=[];counts=Counter();written=[];total=0
    fields=['hgnc_id','gene_symbol','gene_id','annotation_symbol','chrom','start','end','strand','assembly',
            'transcript_id','transcript_version','selection_method','sequence_length','cds_parts','sequence']
    try:
        if not Path(str(fasta)+'.fai').exists():pysam.faidx(str(fasta))
        with pysam.FastaFile(str(fasta)) as genome, (staging/'genes.fa').open('w') as fa, \
             (staging/'genes.csv').open('w',newline='') as meta, (staging/'cds_blocks.bed').open('w') as bed, \
             (staging/'cds_blocks.jsonl').open('w') as block_meta:
            writer=csv.DictWriter(meta,fieldnames=fields);writer.writeheader()
            for gene in genes:
                selected=choose(transcripts.get(gene['gene_id'],{}))
                if not selected:
                    audit.append({'gene_symbol':gene['gene_symbol'],'gene_id':gene['gene_id'],'reason':'no_CDS_at_selected_locus'})
                    continue
                tid,entry=selected
                parts=sorted(entry['parts'])
                if any(parts[i][0]<=parts[i-1][1] for i in range(1,len(parts))):
                    raise ValueError(f'Overlapping CDS parts within transcript {tid}')
                genomic_parts=parts
                if gene['strand']=='-':parts=list(reversed(parts))
                sequences=[];block_rows=[];offset=0
                for index,(start,end,phase) in enumerate(parts):
                    seq=genome.fetch(gene['chrom'],start-1,end)
                    if gene['strand']=='-':seq=reverse_complement(seq)
                    sequences.append(seq)
                    block={'gene_symbol':gene['gene_symbol'],'transcript_id':tid,'block':index,
                           'chrom':gene['chrom'],'genomic_start':start-1,'genomic_end':end,'strand':gene['strand'],
                           'cds_start':offset,'cds_end':offset+len(seq),'phase':phase}
                    block_rows.append(block);offset+=len(seq)
                    bed.write(f"{gene['chrom']}\t{start-1}\t{end}\t{gene['gene_symbol']}\t0\t{gene['strand']}\n")
                    block_meta.write(json.dumps(block)+'\n')
                sequence=''.join(sequences)
                method='MANE_Select' if 'MANE_Select' in entry['tags'] else 'Ensembl_canonical' if 'Ensembl_canonical' in entry['tags'] else 'longest_CDS'
                counts[method]+=1
                if len(sequence)%3:counts['length_not_multiple_of_three']+=1
                if 'cds_start_NF' in entry['tags'] or 'cds_end_NF' in entry['tags']:counts['annotated_partial_CDS']+=1
                row={k:gene.get(k,'') for k in fields}
                row.update(transcript_id=tid,transcript_version=entry['version'],selection_method=method,
                           sequence_length=len(sequence),cds_parts=json.dumps(block_rows,separators=(',',':')),sequence=sequence)
                writer.writerow(row)
                fa.write(f">{gene['gene_symbol']} transcript={tid} type=spliced_CDS\n")
                for i in range(0,len(sequence),80):fa.write(sequence[i:i+80]+'\n')
                total+=len(sequence);written.append({'gene_symbol':gene['gene_symbol'],'length':len(sequence),'transcript_id':tid,
                                                     'tags':sorted(entry['tags'])})
        report={'representation':'One spliced annotated CDS per HGNC gene; MANE Select, then Ensembl canonical, then longest CDS; ties by transcript ID',
                'source_genes':len(genes),'genes_written':len(written),'cds_bases':total,'selection_counts':dict(counts),
                'unresolved':audit,'transcripts':written,
                'coordinates':'BED/genomic and CDS offsets are 0-based half-open; gene start/end fields retain 1-based inclusive genomic locus',
                'phase_policy':'All annotated CDS bases retained; phase is metadata, not trimmed from every exon',
                'stop_codon_policy':'Only GTF CDS features; separate stop_codon features are not appended',
                'limitations':['CDS excludes UTR exons and alternative-only coding exons',
                               'Spliced CDS joins are transcript boundaries, not contiguous WES genomic sequence',
                               'Incomplete annotated CDS retained and counted; no invented bases'],
                'inputs':{str(p):file_hash(p) for p in (fasta,gtf,genes_csv)},
                'outputs':{p.name:file_hash(p) for p in staging.iterdir()}}
        write_json(staging/'report.json',report)
        staging.rename(output)
        return report
    except BaseException:
        shutil.rmtree(staging)
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta',required=True);p.add_argument('--gtf',required=True)
    p.add_argument('--genes',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    r=extract_cds(args.fasta,args.gtf,args.genes,args.output)
    print(json.dumps({k:r[k] for k in ('source_genes','genes_written','cds_bases','selection_counts')},indent=2))


if __name__=='__main__':main()
