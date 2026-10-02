"""Run the fine-tuned reference and held-out WES workflow in isolated stages."""
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from llm_mapper.preparation.downloads import file_hash,write_json

REF=Path('results/gb_dna_300m_exon_infonce_v002_cds_2026-10-01')
WES=Path('results/na19239_gb_dna_300m_exon_infonce_v002_retrieval')
CHECKPOINT=Path('results/finetuning/gbdna300m_exon_infonce_v002/training/checkpoint-001563')
CDS=Path('data/reference/GRCh38-ensembl-110/hgnc_cds')
BASE=Path('results/na19239_gb_dna_300m_retrieval')


def main():
    REF.mkdir(parents=True,exist_ok=True);WES.mkdir(parents=True,exist_ok=True)
    with (REF/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cm=json.loads((CHECKPOINT/'checkpoint_manifest.json').read_text())
        assert file_hash(CHECKPOINT/'model.safetensors')==cm['weight_sha256']
        if not (WES/'selection').exists():shutil.copytree(BASE/'selection',WES/'selection')
        for p in (BASE/'selection').iterdir():
            if p.is_file():assert file_hash(p)==file_hash(WES/'selection'/p.name)
        env=os.environ.copy();env.update(HF_HOME='data/hf-cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR='results/.cache/matplotlib',NUMBA_CACHE_DIR='results/.cache/numba',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',VECLIB_MAXIMUM_THREADS='4')
        config=json.loads((CHECKPOINT/'inference_config.json').read_text())
        provenance=dict(checkpoint=str(CHECKPOINT),weight_sha256=cm['weight_sha256'],reference_results=str(REF),wes_results=str(WES),selection_source=str(BASE/'selection'),sw_analysis=False)
        write_json(REF/'experiment.json',provenance)
        def run(name,module,arguments):
            marker=REF/f'{name}.complete.json'
            if marker.exists():return
            command=[sys.executable,'-u','-m',module,*map(str,arguments)]
            write_json(REF/'pipeline_status.json',dict(status='running',stage=name,updated_at_utc=datetime.now(timezone.utc).isoformat()))
            with (REF/f'{name}.log').open('a') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=log)
            if result.returncode:
                write_json(REF/'pipeline_status.json',dict(status='failed',stage=name,returncode=result.returncode));raise RuntimeError(f'{name} failed; see stage log')
            write_json(marker,dict(status='complete',command=command,completed_at_utc=datetime.now(timezone.utc).isoformat()))
        run('01_reference','llm_mapper.gene_database',['--fasta',CDS/'genes.fa','--metadata',CDS/'genes.csv','--output',REF/'vector_db','--model',CHECKPOINT,'--revision',config['revision'],'--device','mps','--attention','sdpa','--batch-size','2','--genes-per-checkpoint','16'])
        run('02_umap','llm_mapper.gene_umap',[REF/'vector_db',REF/'plots'])
        clusters=REF/'louvain_r0.3_k30'
        run('03_clusters','llm_mapper.gene_clusters',[REF/'vector_db',clusters,'--resolution','.3','--neighbors','30','--top-n','20'])
        with (clusters/'gene_clusters.csv').open() as f:labels=sorted({int(r['cluster']) for r in csv.DictReader(f)})
        if len(labels)>1:
            run('04_cluster_biology','scripts.characterize_gene_clusters',[REF/'vector_db',clusters/'gene_clusters.csv','data/reference/GRCh38-ensembl-110/sources/hgnc_complete_set.tsv',clusters/'gene_analysis','--clusters',*labels,'--top-n','50'])
        run('05_read_embeddings','scripts.evaluate_read_retrieval',['embed',REF/'vector_db',WES/'selection',WES/'embeddings','--device','mps','--batch-size','32'])
        run('06_retrieval','scripts.evaluate_read_retrieval',['evaluate',REF/'vector_db',WES/'selection',WES/'embeddings',REF/'plots',WES/'evaluation','--study-label','Fine-tuned v002'])
        run('07_verification','scripts.verify_read_retrieval',[REF/'vector_db',WES,CDS/'genes.fa',CDS/'cds_blocks.jsonl','--device','cpu'])
        comparison={};table=['# Held-out WES retrieval: base versus fine-tuned encoder','', 'Same 935 reads and candidate universes; no read reselection. The fine-tuned encoder held out all 12 target genes from training and validation.','', '| Model | Universe | Top-1 | Top-5 | Top-50 | Top-100 | Top-250 |','|---|---|---:|---:|---:|---:|---:|']
        for label,root in [('Base',BASE),('Fine-tuned v002',WES)]:
            comparison[label]={}
            for scope in ['all_hgnc','targets_only']:
                m=json.loads((root/'evaluation'/scope/'metrics.json').read_text());comparison[label][scope]=m
                table.append('| '+label+' | '+scope+' | '+' | '.join(f'{m[k]:.2%}' for k in ['accuracy','top5_accuracy','top50_accuracy','top100_accuracy','top250_accuracy'])+' |')
        table+=['','Top-50/100/250 are trivially 100% in the 12-candidate universe. Specificity is conditional on the sampled targets. UMAP is reference-only for each encoder; coordinates are not directly comparable across encoders.','']
        (WES/'evaluation/comparison.md').write_text('\n'.join(table));write_json(WES/'evaluation/comparison.json',comparison)
        write_json(REF/'pipeline_status.json',dict(status='complete',**provenance,completed_at_utc=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':main()
