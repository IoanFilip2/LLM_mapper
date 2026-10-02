"""Streaming sequence input and explicit, lossless-by-default chunking."""
from dataclasses import dataclass, replace
from pathlib import Path
from itertools import islice
import csv
import gzip
import sys


@dataclass(frozen=True)
class SequenceRecord:
    id: str
    sequence: str
    label: str = ""
    start: int = 0
    end: int | None = None
    # Source-read alignment; chunking does not rewrite its CIGAR or coordinates.
    alignment: dict | None = None

    def __post_init__(self):
        sequence = "".join(self.sequence.split()).upper()
        if not sequence or set(sequence) - set("ACGTURYSWKMBDHVN"):
            raise ValueError(f"Empty or invalid nucleotide sequence: {self.id}")
        object.__setattr__(self, "sequence", sequence)
        if self.end is None:
            object.__setattr__(self, "end", self.start + len(sequence))


class SequenceReader:
    """One record at a time; BAM/CRAM need no index and use original read orientation.

    FASTA memory is bounded by the largest record, not the complete file.
    Secondary/supplementary BAM/CRAM alignments are excluded; unmapped reads are kept.
    CSV supports existing gene extraction output (sequence, gene_symbol/gene_id).
    """
    def __init__(self, path, format=None, reference=None):
        self.reference = reference
        self.path = Path(path)
        suffixes = self.path.suffixes
        suffix = suffixes[-2] if suffixes[-1:] == [".gz"] else self.path.suffix
        self.format = format or {".fa": "fasta", ".fna": "fasta", ".fasta": "fasta",
                                 ".fq": "fastq", ".fastq": "fastq", ".bam": "bam",
                                 ".cram": "cram", ".csv": "csv"}.get(suffix.lower())
        if self.format not in {"fasta", "fastq", "bam", "cram", "csv"}:
            raise ValueError(f"Unsupported sequence format: {self.path}")

        if self.format == "cram" and reference is None:
            raise ValueError("CRAM input requires an explicit decoding reference")

    def __iter__(self):
        if self.format in {"bam", "cram"}:
            import pysam
            with pysam.AlignmentFile(str(self.path), "rc" if self.format == "cram" else "rb", check_sq=False,
                                     reference_filename=str(self.reference) if self.reference else None) as handle:
                for read in handle.fetch(until_eof=True):
                    if read.is_secondary or read.is_supplementary or not read.query_sequence:
                        continue
                    mate = "/1" if read.is_read1 else "/2" if read.is_read2 else ""
                    yield SequenceRecord(read.query_name + mate, read.get_forward_sequence(),
                                         alignment={"read_group": read.get_tag("RG") if read.has_tag("RG") else None,
                                                    "flag": read.flag, "contig": read.reference_name,
                                                    "reference_start": read.reference_start,
                                                    "reference_end": read.reference_end, "cigar": read.cigarstring,
                                                    "mapping_quality": read.mapping_quality})
            return
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt") as handle:
            if self.format == "csv":
                csv.field_size_limit(sys.maxsize)
                for i, row in enumerate(csv.DictReader(handle)):
                    label = row.get("gene_symbol") or row.get("gene_id") or row.get("label", "")
                    yield SequenceRecord(row.get("id") or label or str(i), row["sequence"], label)
            elif self.format == "fasta":
                name, parts = None, []
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if name is not None:
                            yield SequenceRecord(name, "".join(parts), name)
                        name, parts = line[1:].split()[0], []
                    elif name is None:
                        raise ValueError("FASTA sequence before header")
                    else:
                        parts.append(line)
                if name is not None:
                    yield SequenceRecord(name, "".join(parts), name)
            else:
                # Wrapped FASTQ is supported; quality characters may include @ and +.
                for header in handle:
                    if not header.startswith("@") or not header[1:].strip():
                        raise ValueError("Invalid FASTQ header")
                    parts = []
                    for line in handle:
                        if line.startswith("+"):
                            break
                        parts.append(line.strip())
                    else:
                        raise ValueError("Truncated FASTQ sequence")
                    sequence = "".join(parts)
                    n_quality = 0
                    while n_quality < len(sequence):
                        quality = handle.readline()
                        if not quality:
                            raise ValueError("Truncated FASTQ quality")
                        n_quality += len(quality.rstrip("\r\n"))
                    if n_quality != len(sequence):
                        raise ValueError("FASTQ sequence/quality length mismatch")
                    yield SequenceRecord(header[1:].split()[0], sequence)


@dataclass(frozen=True)
class ReadChunker:
    length: int | None = None
    stride: int | None = None

    def __post_init__(self):
        if self.length is not None and self.length < 1:
            raise ValueError("Chunk length must be positive")
        if self.stride is not None and (self.length is None or not 1 <= self.stride <= self.length):
            raise ValueError("Stride must be between 1 and chunk length")

    def __call__(self, records):
        for record in records:
            if self.length is None or len(record.sequence) <= self.length:
                yield record
                continue
            for start in range(0, len(record.sequence), self.stride or self.length):
                end = min(start + self.length, len(record.sequence))
                yield replace(record, sequence=record.sequence[start:end],
                              start=record.start + start, end=record.start + end)
                if end == len(record.sequence):
                    break


def batches(records, batch_size):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    iterator = iter(records)
    while batch := list(islice(iterator, batch_size)):
        yield batch
