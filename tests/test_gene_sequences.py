import csv
import sys
import random
import unittest
from pathlib import Path

csv.field_size_limit(sys.maxsize)

GENE_SEQ_CSV = "data/genome/gene_sequences.csv"
N_CHECK = 10  # Number of random rows to check (or adjust as you like)

class TestGeneSequenceLengths(unittest.TestCase):
    @unittest.skipUnless(Path(GENE_SEQ_CSV).exists(), "External gene sequence fixture not present")
    def test_length_consistency(self):
        with open(GENE_SEQ_CSV) as f:
            reader = list(csv.DictReader(f))
        # You might prefer random.sample, or just check the first N
        sample_rows = random.sample(reader, min(N_CHECK, len(reader)))
        for row in sample_rows:
            start = int(row['start'])
            end = int(row['end'])
            seq = row['sequence']
            chrom = row['chrom']
            gene = row['gene_symbol']
            expected_len = end - start + 1
            actual_len = len(seq)
            self.assertEqual(
                expected_len, actual_len,
                f"Length mismatch for gene {gene} ({chrom}:{start}-{end}): expected {expected_len}, got {actual_len}"
            )

class SmokeTest(unittest.TestCase):
    def test_sanity(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
