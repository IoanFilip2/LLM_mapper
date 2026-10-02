"""Benchmark exact SW variants and worker counts on stratified actual graph pairs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json,time
from pathlib import Path
import numpy as np
from scipy import sparse
from llm_mapper.alignment import SmithWaterman,SWConfig
from llm_mapper.neighborhood_alignment import load_inputs


def main():
    root=Path('results/gb_dna_300m_cds_2026-09-28')
    db,a,seqs,nodes,xy,signature=load_inputs(root/'vector_db',root/'louvain_r0.3_k30',
        'data/reference/GRCh38-ensembl-110/hgnc_cds/genes.fa')
    upper=sparse.triu(a,k=1).tocoo();length=np.array([len(s) for s in seqs],dtype=np.int64)
    costs=length[upper.row]*length[upper.col];order=np.argsort(costs)
    ids=np.unique(np.r_[order[np.linspace(0,len(order)-1,32,dtype=int)],order[-4:]])
    pairs=[(seqs[int(upper.row[i])],seqs[int(upper.col[i])]) for i in ids]
    reports=[];baseline=None
    for algorithm in ['striped','scan']:
        model=SmithWaterman(SWConfig(algorithm=algorithm));model.compare('ACGT'*100,'ACGT'*100)
        for workers in [1,4,8]:
            start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(lambda p:model.compare(*p),pairs))
            seconds=time.perf_counter()-start
            scores=[r['score'] for r in results]
            if baseline is None:baseline=scores
            assert scores==baseline,'SW implementations or threading disagree'
            row={'algorithm':algorithm,'workers':workers,'pairs':len(pairs),'seconds':seconds,
                 'dp_cells_both_strands':int(2*costs[ids].sum()),'cell_updates_per_second':float(2*costs[ids].sum()/seconds)}
            print(json.dumps(row),flush=True);reports.append(row)
    out=root/'sw_neighborhood';out.mkdir(exist_ok=True)
    (out/'benchmark.json').write_text(json.dumps({'reports':reports,'scoring':asdict(SWConfig()),'input_signature':signature,
        'note':'32 pair-cost quantiles plus the four most expensive pairs, unique union; one timed pass per setting after warmup. All raw scores agree exactly.'},indent=2)+'\n')


if __name__=='__main__':main()
