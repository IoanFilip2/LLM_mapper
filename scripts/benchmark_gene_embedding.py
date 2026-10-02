"""Benchmark actual reference windows and validate reduced precision before a long run."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import time
import numpy as np
import pysam
import torch
from llm_mapper.embedders import EmbeddingConfig, create_embedder
from llm_mapper.devices import synchronize
from llm_mapper.benchmark import MemorySampler
from llm_mapper.index import normalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--revision', default='a833c5602b0dad0728c0aeeecaf43086ebeb96ca')
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not Path(args.fasta + '.fai').exists():
        pysam.faidx(args.fasta)
    rng = random.Random(42)
    with pysam.FastaFile(args.fasta) as f:
        names = [n for n, length in zip(f.references, f.lengths) if length >= 250]
        selected = rng.sample(names, 128)
        windows = []
        for name in selected:
            offset = rng.randrange(f.get_reference_length(name) - 250 + 1)
            windows.append(f.fetch(name, offset, offset + 250))
    validation = [s[:length] for length in [100, 150, 250] for s in windows[:8]]
    config = EmbeddingConfig(revision=args.revision)
    reports, comparisons = [], []
    cpu_vectors = None
    for device, dtype, batch_sizes in [('cpu', 'float32', [4]), ('mps', 'float32', [4, 16]),
                                       ('mps', 'float16', [4, 16, 32, 64, 128])]:
        print(f'Loading {device} {dtype}', flush=True)
        embedder = create_embedder(replace(config, dtype=dtype), device)
        vectors = embedder.encode(validation, batch_size=4)
        if cpu_vectors is None:
            cpu_vectors = vectors
        cosines = (normalize(cpu_vectors) * normalize(vectors)).sum(axis=1)
        relative = np.linalg.norm(vectors-cpu_vectors, axis=1)/np.linalg.norm(cpu_vectors, axis=1)
        passed = bool(cosines.min() >= .999 and relative.max() <= .01)
        comparisons.append({'device': device, 'dtype': dtype, 'minimum_cosine_to_cpu_float32': float(cosines.min()),
                            'maximum_relative_l2_to_cpu_float32': float(relative.max()), 'passed': passed,
                            'validation_lengths': [100, 150, 250], 'sequences_per_length': 8})
        if not passed:
            print('Numerical validation failed; not benchmarking this configuration', flush=True)
            continue
        for size in batch_sizes:
            sequences = windows[:32] if device == 'cpu' else windows
            embedder.encode(sequences[:size], batch_size=size)
            repeats = []
            with MemorySampler(device, interval=.05) as memory:
                for repeat in range(2):
                    synchronize(device)
                    start = time.perf_counter()
                    result = embedder.encode(sequences, batch_size=size)
                    synchronize(device)
                    elapsed = time.perf_counter()-start
                    repeats.append(elapsed)
            row = {'device': device, 'dtype': dtype, 'batch_size': size, 'sequences': len(sequences),
                   'window_length': 250, 'repeat_seconds': repeats,
                   'windows_per_second': len(sequences)/float(np.median(repeats)),
                   'sampled_peak_rss_bytes': memory.peak_rss,
                   'sampled_peak_device_bytes': memory.peak_device}
            reports.append(row)
            print(json.dumps(row), flush=True)
        del embedder
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    best = max((r for r in reports if r['device']=='mps'), key=lambda r:r['windows_per_second'])
    result = {'purpose': 'Throughput and numerical checks on real gene windows; not mapping accuracy',
              'seed': 42, 'model_config': config.__dict__, 'comparisons': comparisons, 'reports': reports,
              'selected': best, 'estimated_hours_for_5285180_windows': 5285180/best['windows_per_second']/3600,
              'memory_note': 'RSS and MPS driver allocations overlap; do not sum. Sample interval 50 ms.',
              'timing_note': 'Two timed repeats after warmup; model loading excluded. Full run may be slower under sustained load.'}
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
