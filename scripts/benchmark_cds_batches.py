"""Batch-size check with matched-length contexts, avoiding padding confounding."""
import gc,json,time
from pathlib import Path
import numpy as np
import pysam,torch
from llm_mapper.embedders import EmbeddingConfig,create_embedder
from llm_mapper.devices import synchronize
from llm_mapper.benchmark import MemorySampler
from llm_mapper.index import normalize


def main():
    root=Path('results/gb_dna_300m_cds_2026-09-28')
    reference='data/reference/GRCh38-ensembl-110/hgnc_cds/genes.fa'
    config=EmbeddingConfig(revision='a833c5602b0dad0728c0aeeecaf43086ebeb96ca',attention='sdpa')
    model=create_embedder(config,'mps')
    reports=[]
    with pysam.FastaFile(reference) as f:
        names=[n for n,length in zip(f.references,f.lengths) if length>=3998][:32]
        for length in [1500,3998]:
            seqs=[f.fetch(n,0,length) for n in names]
            baseline=None
            for batch in [2,4,8,16]:
                model.encode(seqs[:batch],batch_size=batch)
                synchronize('mps');torch.mps.empty_cache()
                times=[]
                with MemorySampler('mps',interval=.05) as memory:
                    for _ in range(2):
                        synchronize('mps');t=time.perf_counter()
                        vectors=model.encode(seqs,batch_size=batch)
                        synchronize('mps');times.append(time.perf_counter()-t)
                if baseline is None:baseline=vectors
                cosine=(normalize(vectors)*normalize(baseline)).sum(axis=1)
                relative=np.linalg.norm(vectors-baseline,axis=1)/np.linalg.norm(baseline,axis=1)
                row={'length':length,'batch_size':batch,'sequences':len(seqs),'repeat_seconds':times,
                     'bases_per_second':length*len(seqs)/float(np.median(times)),
                     'minimum_cosine_to_batch2':float(cosine.min()),'maximum_relative_l2_to_batch2':float(relative.max()),
                     'passed':bool(cosine.min()>=.99999 and relative.max()<.001),
                     'sampled_peak_rss_bytes':memory.peak_rss,'sampled_peak_device_bytes':memory.peak_device}
                reports.append(row);print(json.dumps(row),flush=True)
    (root/'matched_length_benchmark.json').write_text(json.dumps({'config':config.__dict__,'reports':reports,
        'note':'32 actual CDS prefixes per length; no padding between sequences. Warmup then two synchronized repeats. RSS and driver memory overlap.'},indent=2)+'\n')


if __name__=='__main__':main()
