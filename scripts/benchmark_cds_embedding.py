"""Validate GB SDPA and benchmark real CDS contexts before whole-reference embedding."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time
import numpy as np
import pysam
import torch
from llm_mapper.embedders import EmbeddingConfig,create_embedder
from llm_mapper.devices import synchronize
from llm_mapper.benchmark import MemorySampler
from llm_mapper.index import normalize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    if not Path(args.fasta+'.fai').exists():pysam.faidx(args.fasta)
    with pysam.FastaFile(args.fasta) as f:
        contexts=[(n,s,min(s+3998,length)) for n,length in zip(f.references,f.lengths) for s in range(0,length,3998)]
        contexts.sort(key=lambda r:(r[2]-r[1],r[0],r[1]))
        positions=np.linspace(0,len(contexts)-1,24).round().astype(int)
        samples=[f.fetch(*contexts[i]) for i in positions]
        long_name=next(n for n,length in zip(f.references,f.lengths) if length>=3998)
        validation=[f.fetch(long_name,0,length) for length in [100,150,250,1024,3998]]
    base=EmbeddingConfig(revision='a833c5602b0dad0728c0aeeecaf43086ebeb96ca',attention='eager')
    cpu=create_embedder(base,'cpu')
    start=time.perf_counter();reference=cpu.encode(validation,batch_size=1);cpu_seconds=time.perf_counter()-start
    del cpu;gc.collect()
    checks=[];reports=[]
    for attention in ['eager','sdpa']:
        print('Validating MPS '+attention,flush=True)
        model=create_embedder(replace(base,attention=attention),'mps')
        values=model.encode(validation,batch_size=1)
        cosine=(normalize(values)*normalize(reference)).sum(axis=1)
        relative=np.linalg.norm(values-reference,axis=1)/np.linalg.norm(reference,axis=1)
        passed=bool(cosine.min()>=.99999 and relative.max()<.001)
        checks.append({'attention':attention,'minimum_cosine_to_cpu_eager':float(cosine.min()),
                       'maximum_relative_l2':float(relative.max()),'passed':passed})
        if not passed:raise ValueError('SDPA numerical validation failed')
        sizes=[1] if attention=='eager' else [1,2,4,8]
        for batch in sizes:
            model.encode([samples[-1]]*batch,batch_size=batch)
            synchronize('mps');torch.mps.empty_cache()
            times=[]
            with MemorySampler('mps',interval=.05) as memory:
                for _ in range(2):
                    synchronize('mps');start=time.perf_counter()
                    values=model.encode(samples,batch_size=batch)
                    synchronize('mps');times.append(time.perf_counter()-start)
            row={'attention':attention,'dtype':'float32','device':'mps','batch_size':batch,
                 'contexts':len(samples),'bases':sum(map(len,samples)),'repeat_seconds':times,
                 'bases_per_second':sum(map(len,samples))/float(np.median(times)),
                 'sampled_peak_rss_bytes':memory.peak_rss,'sampled_peak_device_bytes':memory.peak_device}
            reports.append(row);print(json.dumps(row),flush=True)
        del model;gc.collect();torch.mps.empty_cache()
    best=max(reports,key=lambda r:r['bases_per_second'])
    report={'purpose':'CDS-context inference performance and CPU/MPS numerical equivalence; not mapping accuracy',
            'config':base.__dict__,'validation_lengths':[100,150,250,1024,3998],
            'cpu_eager_validation_seconds':cpu_seconds,'checks':checks,'reports':reports,'selected':best,
            'total_contexts':len(contexts),'context_length_quantiles':np.quantile([e-s for _,s,e in contexts],[0,.5,.9,1]).tolist(),
            'memory_note':'RSS and MPS driver allocation overlap; do not sum. 50-ms sampling; model load excluded.',
            'sampling_note':'24 length-quantile contexts; throughput is indicative, not a whole-reference runtime guarantee.'}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
