"""Audit a completed gene database without loading a foundation model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import numpy as np
from .gene_database import GeneVectorDatabase, _digest
from .preparation.downloads import file_hash, write_json


def verify_database(path):
    root = Path(path)
    db = GeneVectorDatabase(root)
    signature = db.manifest['signature']
    matrices = {}
    checks = {}
    for level, count in [('gene', signature['genes']), ('window', signature['windows'])]:
        source = root / f'{level}_vectors.npy'
        values = np.load(source, mmap_mode='r', allow_pickle=False)
        if values.shape != (count, signature['dimension']) or values.dtype != np.float32:
            raise ValueError(f'Unexpected {level} matrix shape/dtype')
        minimum, maximum = float('inf'), 0.
        for start in range(0, count, 4096):
            block = values[start:start+4096]
            norms = np.linalg.norm(block, axis=1)
            if not np.isfinite(block).all() or not np.allclose(norms, 1., atol=1e-5):
                raise ValueError(f'Invalid {level} vectors at row {start}')
            minimum, maximum = min(minimum, float(norms.min())), max(maximum, float(norms.max()))
        matrices[level] = values
        checks[level] = {'shape': list(values.shape), 'dtype': str(values.dtype),
                         'norm_min': minimum, 'norm_max': maximum, 'sha256': file_hash(source)}
    if len(db.genes) != signature['genes'] or len({r['gene_symbol'] for r in db.genes}) != len(db.genes):
        raise ValueError('Gene metadata count/uniqueness mismatch')
    cursor = 0
    for index, row in enumerate(db.genes):
        expected = (row['sequence_length'] + signature['chunk_length'] - 1) // signature['chunk_length']
        if row['gene_index'] != index or row['window_first'] != cursor or row['window_count'] != expected:
            raise ValueError('Metadata coverage mismatch')
        cursor += expected
    if cursor != signature['windows']:
        raise ValueError('Context count mismatch')
    with sqlite3.connect(root / 'metadata.sqlite') as sql:
        if sql.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite integrity check failed')
        blocks = list(sql.execute('SELECT * FROM blocks ORDER BY block_id'))
    gene_cursor = window_cursor = 0
    for index, (block_id, a, b, c, d, gh, wh, seconds) in enumerate(blocks):
        if block_id != index or a != gene_cursor or c != window_cursor or b <= a or d <= c:
            raise ValueError('Checkpoint coverage mismatch')
        if _digest(matrices['gene'][a:b]) != gh or _digest(matrices['window'][c:d]) != wh:
            raise ValueError('Checkpoint checksum mismatch')
        gene_cursor, window_cursor = b, d
    if (gene_cursor, window_cursor) != (signature['genes'], signature['windows']):
        raise ValueError('Incomplete checkpoint coverage')
    sample = np.unique(np.linspace(0, len(db.genes)-1, min(12, len(db.genes)), dtype=int))
    hits = db.search_vectors(matrices['gene'][sample], k=1)
    if any(row[0]['cosine_similarity'] < .99999 for row in hits):
        raise ValueError('Exact cosine self-query check failed')
    try:
        database_label = root.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        database_label = root.name
    return {'status': 'passed', 'verified_at': datetime.now(timezone.utc).isoformat(),
            'database': database_label, 'model_fingerprint': signature['model_fingerprint'],
            'matrices': checks, 'checkpoint_blocks_verified': len(blocks),
            'coding_bases': sum(r['sequence_length'] for r in db.genes),
            'sqlite_integrity': 'ok', 'self_queries': len(sample),
            'self_query_note': 'Top similarity checked; identical vectors can tie across genes. This is storage/retrieval validation, not biological accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = verify_database(args.database)
    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
