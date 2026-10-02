"""Resumable gene/CDS context embeddings using any SequenceEmbedder.

SQLite stores labeled gene ranges and committed blocks; NumPy matrices store
normalized vectors. A block commit follows durable vector writes. Incomplete
blocks are overwritten on resume and cannot be queried as a complete database.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
from .embedders import EmbeddingConfig, create_embedder
from .index import normalize
from .preparation.downloads import file_hash, write_json


def catalog_from_csv(path, chunk_length):
    csv.field_size_limit(sys.maxsize)
    rows, seen, offset = [], set(), 0
    with open(path) as handle:
        for source in csv.DictReader(handle):
            sequence = source.pop('sequence', None)
            if sequence is None:
                raise ValueError('Gene metadata CSV must include sequence for catalog validation')
            symbol = source['gene_symbol']
            if symbol in seen:
                raise ValueError(f'Duplicate gene symbol: {symbol}')
            seen.add(symbol)
            length = len(sequence)
            if not length or length != int(source['sequence_length']):
                raise ValueError(f'Invalid sequence length: {symbol}')
            count = math.ceil(length/chunk_length)
            rows.append({**source, 'gene_index': len(rows), 'sequence_length': length,
                         'window_first': offset, 'window_count': count,
                         'gc_fraction': (sequence.count('G')+sequence.count('C'))/length,
                         'n_fraction': sequence.count('N')/length})
            offset += count
    # Deterministic length ordering reduces padding within embedding batches.
    rows.sort(key=lambda r:(r['sequence_length'],r['gene_symbol']))
    offset=0
    for index,row in enumerate(rows):
        row['gene_index']=index;row['window_first']=offset;offset+=row['window_count']
    return rows


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_gene_database(fasta, metadata_csv, embedder, output, chunk_length=3998,
                        batch_size=16, genes_per_checkpoint=16, max_blocks=None):
    import pysam
    if min(chunk_length, batch_size, genes_per_checkpoint) < 1:
        raise ValueError('Chunk length, batch size and checkpoint size must be positive')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent writers, including across CLI invocations.
    import fcntl
    lock = (root/'writer.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another writer holds this database lock')
    try:
        return _build(fasta, metadata_csv, embedder, root, chunk_length,
                      batch_size, genes_per_checkpoint, max_blocks)
    finally:
        lock.close()


def _build(fasta, metadata_csv, embedder, root, chunk_length, batch_size, block_size, max_blocks):
    import pysam
    started = time.perf_counter()
    rows = catalog_from_csv(metadata_csv, chunk_length)
    if not rows:
        raise ValueError('No genes in metadata')
    if not Path(str(fasta)+'.fai').exists():
        pysam.faidx(str(fasta))
    if not getattr(embedder, 'dimension', None):
        with pysam.FastaFile(str(fasta)) as reference:
            probe_sequence=reference.fetch(rows[0]['gene_symbol'],0,min(chunk_length,rows[0]['sequence_length']))
        probe=normalize(embedder.encode([probe_sequence],batch_size=1))
        if probe.shape[0] != 1 or probe.shape[1] == 0:
            raise ValueError('Cannot infer embedding dimension')
        embedder.dimension=probe.shape[1]
    n_genes = len(rows)
    n_windows = rows[-1]['window_first']+rows[-1]['window_count']
    signature = {'format_version': 1, 'model_signature': embedder.signature,
                 'model_fingerprint': embedder.fingerprint, 'model_config': asdict(embedder.config),
                 'fasta_sha256': file_hash(fasta), 'metadata_sha256': file_hash(metadata_csv),
                 'chunk_length': chunk_length, 'stride': chunk_length, 'genes_per_checkpoint': block_size,
                 'pooling': 'base-length-weighted mean of raw window embeddings, then L2 normalization',
                 'window_vectors': 'L2-normalized raw model embeddings',
                 'storage_dtype': 'float32', 'ordering': 'sequence_length_then_symbol',
                 'representation': 'spliced_CDS' if 'cds_parts' in rows[0] else 'full_genomic_gene_span',
                 'orientation': 'gene/transcript 5-prime to 3-prime',
                 'genes': n_genes, 'windows': n_windows, 'dimension': embedder.dimension}
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing['signature'] != signature:
            raise ValueError('Resume signature mismatch: model, sources or aggregation changed')
    else:
        if (root/'metadata.sqlite').exists():
            raise FileExistsError('Untracked database exists; choose a new output directory')
        existing = {'signature': signature, 'status': 'building', 'created_at': datetime.now(timezone.utc).isoformat(),
                    'device': embedder.device, 'batch_size': batch_size, 'completed_genes': 0, 'completed_windows': 0}
        write_json(manifest_path, existing)
    if not Path(str(fasta)+'.fai').exists():
        pysam.faidx(str(fasta))
    database = sqlite3.connect(root/'metadata.sqlite')
    database.execute('PRAGMA synchronous=FULL')
    database.execute('CREATE TABLE IF NOT EXISTS genes (gene_index INTEGER PRIMARY KEY, gene_symbol TEXT UNIQUE, metadata TEXT NOT NULL)')
    database.execute('CREATE TABLE IF NOT EXISTS blocks (block_id INTEGER PRIMARY KEY, first_gene INTEGER, stop_gene INTEGER, first_window INTEGER, stop_window INTEGER, gene_sha256 TEXT, window_sha256 TEXT, seconds REAL)')
    database.executemany('INSERT OR IGNORE INTO genes VALUES (?,?,?)',
                         [(r['gene_index'],r['gene_symbol'],json.dumps(r)) for r in rows])
    database.commit()
    def matrix(name, shape):
        path = root/name
        if path.exists():
            value = np.load(path, mmap_mode='r+', allow_pickle=False)
            if value.shape != shape or value.dtype != np.float32:
                raise ValueError(f'Invalid stored matrix: {path}')
            return value
        return np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
    genes = matrix('gene_vectors.npy', (n_genes, embedder.dimension))
    windows = matrix('window_vectors.npy', (n_windows, embedder.dimension))
    committed = {r[0]: r for r in database.execute('SELECT * FROM blocks ORDER BY block_id')}
    # Revalidate committed vector bytes once on resume; incomplete bytes are overwritten.
    for block in committed.values():
        _, a,b,c,d,gh,wh,_ = block
        if _digest(genes[a:b]) != gh or _digest(windows[c:d]) != wh:
            raise ValueError('Committed checkpoint checksum mismatch')
    if existing['status'] != 'complete':
        history = existing.setdefault('execution_history', [])
        if not history and committed:
            history.append({'started_at': existing['created_at'], 'device': existing['device'],
                            'batch_size': existing['batch_size'], 'resume_gene': 0,
                            'note': 'Original invocation, recorded retrospectively on resume'})
        history.append({'started_at': datetime.now(timezone.utc).isoformat(),
                        'device': embedder.device, 'batch_size': batch_size,
                        'resume_gene': max((b[2] for b in committed.values()), default=0)})
        existing.update(device=embedder.device, batch_size=batch_size)
        write_json(manifest_path, existing)
    completed_this_run = 0
    with pysam.FastaFile(str(fasta)) as genome:
        if set(genome.references) != {r['gene_symbol'] for r in rows}:
            raise ValueError('FASTA and metadata gene symbols differ')
        for block_id, first_gene in enumerate(range(0,n_genes,block_size)):
            stop_gene = min(first_gene+block_size,n_genes)
            if block_id in committed:
                continue
            block_start = time.perf_counter()
            first_window = rows[first_gene]['window_first']
            last = rows[stop_gene-1]
            stop_window = last['window_first']+last['window_count']
            sums = np.zeros((stop_gene-first_gene,embedder.dimension),dtype=np.float64)
            lengths = np.zeros(stop_gene-first_gene,dtype=np.int64)
            sequences, ids, weights = [], [], []
            cursor = first_window
            def flush_batch():
                nonlocal cursor
                if not sequences:
                    return
                raw = np.asarray(embedder.encode(sequences,batch_size=batch_size),dtype=np.float32)
                if raw.shape != (len(sequences),embedder.dimension):
                    raise ValueError('Invalid embedding batch shape')
                vectors = normalize(raw)
                windows[cursor:cursor+len(sequences)] = vectors
                np.add.at(sums, ids, raw.astype(np.float64)*np.asarray(weights)[:,None])
                np.add.at(lengths, ids, weights)
                cursor += len(sequences)
                sequences.clear(); ids.clear(); weights.clear()
            for gene_index in range(first_gene,stop_gene):
                row = rows[gene_index]
                sequence = genome.fetch(row['gene_symbol'])
                if len(sequence) != row['sequence_length']:
                    raise ValueError('FASTA/metadata length mismatch')
                for offset in range(0,len(sequence),chunk_length):
                    piece = sequence[offset:offset+chunk_length]
                    sequences.append(piece); ids.append(gene_index-first_gene); weights.append(len(piece))
                    if len(sequences)==batch_size:
                        flush_batch()
            flush_batch()
            if cursor != stop_window or list(lengths) != [r['sequence_length'] for r in rows[first_gene:stop_gene]]:
                raise ValueError('Window/base coverage mismatch')
            genes[first_gene:stop_gene] = normalize(sums/lengths[:,None])
            windows.flush(); genes.flush()
            for name in ('window_vectors.npy','gene_vectors.npy'):
                with (root/name).open('rb') as handle:
                    os.fsync(handle.fileno())
            seconds = time.perf_counter()-block_start
            database.execute('INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?)',
                             (block_id,first_gene,stop_gene,first_window,stop_window,
                              _digest(genes[first_gene:stop_gene]),_digest(windows[first_window:stop_window]),seconds))
            database.commit()
            existing.update(status='building',completed_genes=stop_gene,completed_windows=stop_window,
                            updated_at=datetime.now(timezone.utc).isoformat())
            write_json(manifest_path,existing)
            print(f'Committed {stop_gene:,}/{n_genes:,} genes; {stop_window:,}/{n_windows:,} windows; block {seconds:.1f}s',flush=True)
            completed_this_run += 1
            if max_blocks is not None and completed_this_run >= max_blocks:
                database.close()
                return existing
    existing.update(status='complete',completed_genes=n_genes,completed_windows=n_windows,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    embedding_block_seconds=sum(r[0] for r in database.execute('SELECT seconds FROM blocks')),
                    last_invocation_seconds=time.perf_counter()-started)
    # Export a compatible gene-centroid index for the existing mapping API.
    index = root/'gene_index'
    index.mkdir(exist_ok=True)
    if not (index/'0.npy').exists():
        os.link(root/'gene_vectors.npy', index/'0.npy')
    with (index/'0.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps({'id':row['gene_symbol'],'label':row['gene_symbol'],
                                     'start':0,'end':row['sequence_length']})+'\n')
    write_json(index/'manifest.json', {'format_version':1,'signature':embedder.signature,
               'fingerprint':embedder.fingerprint,'config':asdict(embedder.config),
               'chunker':{'length':chunk_length,'stride':chunk_length},'count':n_genes,
               'dimension':embedder.dimension,'shards':1,'metric':'cosine','device':embedder.device,
               'representation':signature['pooling']})
    database.close()
    write_json(manifest_path,existing)
    return existing


class GeneVectorDatabase:
    def __init__(self,path,embedder=None):
        self.path=Path(path)
        self.manifest=json.loads((self.path/'manifest.json').read_text())
        if self.manifest['status']!='complete':
            raise ValueError('Database is incomplete; resume embedding first')
        if embedder and embedder.fingerprint != self.manifest['signature']['model_fingerprint']:
            raise ValueError('Reference/read model fingerprint mismatch')
        self.embedder=embedder
        with sqlite3.connect(self.path/'metadata.sqlite') as db:
            self.genes=[json.loads(r[0]) for r in db.execute('SELECT metadata FROM genes ORDER BY gene_index')]
        self.first=np.array([r['window_first'] for r in self.genes])

    def search_vectors(self,query,k=10,level='gene',block_size=4096):
        if level not in {'gene','window'} or k<1 or block_size<1:
            raise ValueError('Invalid search parameters')
        query=normalize(query)
        matrix=np.load(self.path/f'{level}_vectors.npy',mmap_mode='r',allow_pickle=False)
        if query.shape[1]!=matrix.shape[1]:
            raise ValueError('Vector dimension mismatch')
        k=min(k,len(matrix))
        result=[]
        for vector in query:
            scores=np.empty(0,dtype=np.float32); indices=np.empty(0,dtype=np.int64)
            for start in range(0,len(matrix),block_size):
                candidates=matrix[start:start+block_size]@vector
                local=np.argpartition(candidates,-min(k,len(candidates)))[-k:]
                scores=np.concatenate([scores,candidates[local]])
                indices=np.concatenate([indices,start+local])
                keep=np.argsort(-scores,kind='stable')[:k]
                scores,indices=scores[keep],indices[keep]
            matches=[]
            for index,score in zip(indices,scores):
                gene_index=int(index) if level=='gene' else int(np.searchsorted(self.first,index,side='right')-1)
                row=self.genes[gene_index]
                start=0 if level=='gene' else (int(index)-row['window_first'])*self.manifest['signature']['chunk_length']
                end=row['sequence_length'] if level=='gene' else min(start+self.manifest['signature']['chunk_length'],row['sequence_length'])
                matches.append({'gene_symbol':row['gene_symbol'],'vector_index':int(index),
                                'gene_sequence_start':start,'gene_sequence_end':end,
                                'cosine_similarity':float(np.clip(score,-1,1))})
            result.append(matches)
        return result

    def search(self,sequences,k=10,level='gene',batch_size=16):
        if self.embedder is None:
            raise ValueError('An embedder is required for sequence queries')
        return self.search_vectors(self.embedder.encode(sequences,batch_size=batch_size),k,level)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta',required=True);p.add_argument('--metadata',required=True);p.add_argument('--output',required=True)
    p.add_argument('--backend',choices=['gb','hf','evo2'],default='gb')
    p.add_argument('--model',default='genbio-ai/GB.DNA-300M');p.add_argument('--revision',help='Pinned revision; defaults to the validated GB checkpoint or main for another model')
    p.add_argument('--device',default='mps');p.add_argument('--dtype',default='float32')
    p.add_argument('--attention',choices=['eager','sdpa'],default='eager')
    p.add_argument('--pooling',default='mean');p.add_argument('--max-length',type=int,default=4000)
    p.add_argument('--kmer',type=int,default=0);p.add_argument('--layer');p.add_argument('--loader')
    p.add_argument('--chunk-length',type=int,default=3998);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--genes-per-checkpoint',type=int,default=16)
    p.add_argument('--plot-output',help='After successful full embedding, generate a gene UMAP here')
    args=p.parse_args(argv)
    args.revision=args.revision or ('a833c5602b0dad0728c0aeeecaf43086ebeb96ca' if args.model=='genbio-ai/GB.DNA-300M' else 'main')
    embedder=create_embedder(EmbeddingConfig(**{k:getattr(args,k) for k in EmbeddingConfig.__dataclass_fields__}),args.device)
    build_gene_database(args.fasta,args.metadata,embedder,args.output,args.chunk_length,args.batch_size,args.genes_per_checkpoint)
    from .verify_gene_database import verify_database
    write_json(Path(args.output)/'verification.json', verify_database(args.output))
    if args.plot_output:
        del embedder
        import gc
        gc.collect()
        if args.device == 'mps':
            import torch
            torch.mps.empty_cache()
        from .gene_umap import plot_database
        plot_database(args.output,args.plot_output)


if __name__=='__main__':
    main()
