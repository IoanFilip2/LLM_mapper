# NA19239: direct CRAM verification workflow

The canonical input is the downloaded IGSR NA19239 GRCh38DH exome CRAM.
All alignment decoding, reference indexing and alignment integrity checks use
**pysam** and its bundled samtools/HTSlib. No standalone samtools executable or
intermediate BAM is required. NA20314 remains deferred for reproducibility;
HG00419 remains deferred for the 250-bp WGS evaluation.

## Completed verification — 2026-09-28

**PASS:** 104,839,877 alignment records decoded through EOF; 93,288,902 primary
sequence-bearing records, all exactly 100 bp. All 3,366 reference contigs match
header lengths and MD5s. Sample identity matches NA19239. No primary sequences
or qualities are missing; no sequence/quality-length mismatches were found.

See the [dedicated QC report](NA19239_qc.md) and [complete JSON](NA19239_qc.json)
for every measured statistic, distributions, reference checks and input hashes.
The total alignment-record count exactly matches the two archived BAS rows.
The report separately labels those upstream statistics; they are not all
recomputed metrics. Duplicate flags affect 28.0008% of primary records, and
13.1979% of mapped primary records have MAPQ 0. These remain in the data and
need explicit treatment when designing the mapping study.

The full scan took 13.44 minutes including reference verification, excluding
download and subsequent provenance hashing. No BAM or bulk FASTQ derivative
was created. A separate 1,000-record reader integration check succeeded; its
local `reader_smoke_test.json` is explicitly prefix-only. Code validation:
41 tests passed and one external legacy-data test was skipped.

## Reproduce

From the repository root:

```sh
.venv/bin/python -m llm_mapper.preparation samples \
  configs/wes_100bp_1000genomes.json --sample NA19239
```

This verifies cached download checksums, acquires the exact decoding reference
if needed, indexes that FASTA and scans NA19239 directly. It writes
`data/wes_100bp_1000genomes/NA19239/qc.json`. Existing samples are not treated as
verified merely because files exist. `--download-only` skips QC; `--export-bam`
optionally writes a BAM derivative but QC still operates on the source CRAM.

To inspect the existing CRAM independently:

```sh
.venv/bin/python -m llm_mapper.preparation inspect \
  data/wes_100bp_1000genomes/NA19239/source/NA19239.alt_bwamem_GRCh38DH.20150826.YRI.exome.cram \
  --reference data/reference/GRCh38-IGSR-decoding/GRCh38_full_analysis_set_plus_decoy_hla.fa \
  --expected-length 100 \
  --output data/wes_100bp_1000genomes/NA19239/inspection.json
```

The manifest workflow additionally verifies sample identity, records input
hashes and computes the combined readiness gate. The independent `inspect`
command provides measurements without asserting the expected sample identity.

To regenerate the shareable report from the completed local QC JSON:

```sh
.venv/bin/python scripts/report_alignment_qc.py \
  data/wes_100bp_1000genomes/NA19239/qc.json \
  --sample NA19239 --output docs/NA19239_qc
```

## Verification stages

1. Validate CRAM, CRAI and BAS download receipts and published MD5s. Record local
   SHA-256 hashes. The original [download report](na19239_download_report.json)
   records acquisition provenance.
2. Fetch and index the exact IGSR full-analysis-set plus decoy/HLA FASTA. The
   reviewed manifest uses the public 1000 Genomes S3 mirror and retains the
   original EBI URL. The Ensembl primary-assembly gene reference is a separate
   artifact and is not substituted for this decoding reference.
3. Call `pysam.quickcheck` for header/EOF integrity. This is an initial check,
   not an internal-data scan.
4. Compare every CRAM `@SQ` name and length with the local FASTA; compute each
   reference sequence MD5 in bounded-size chunks and compare with header `M5`.
   A mismatch stops processing. The readiness gate requires all header MD5s
   to be present and match.
5. Use `pysam.AlignmentFile(..., "rc", reference_filename=...)` and
   `fetch(until_eof=True)` to decode the complete file. Sequence and quality
   fields are accessed for every record, including secondary/supplementary
   records. Exceptions abort the run; a successful full-file report requires
   reaching EOF without a read limit.
6. Require the expected sample in every read-group `SM`, the GRCh38 primary
   chromosome length checks, full decoding, reference verification and at least
   95% of primary sequence-bearing reads at the expected 100-bp length.
7. When BAS statistics are supplied, require the observed total alignment-record
   count to equal the sum of the archived BAS read-group totals.

Any positive `--limit` makes the result provisional, even if the file contains
fewer records than that limit. Full-file QC cannot be inferred from prefix probes.

## Statistics and denominators

The machine-readable report retains all measured statistics:

- Total alignment records; secondary and supplementary flag counts (these may
  overlap if both flags are present); all-record QC-failed count.
- Primary records: duplicates, unmapped, QC-failed, paired, proper-pair flag,
  read1/read2, reverse strand and mate-unmapped counts. These counts include
  primary records without sequence. Flags describe archived alignments; no
  duplicate detection or re-alignment is performed.
- Primary sequence-bearing records: complete length histogram, modal length,
  exact expected-length fraction and per-read-group length histograms.
- Missing primary sequences; missing qualities among sequence-bearing primary
  reads; sequence/quality-length mismatches.
- Primary mapped records: full MAPQ histogram and counts by reference contig.
  MAPQ 255 means unavailable and is not treated as high confidence.
- Primary-record clipping: number of reads and bases with soft and hard clips.
  Soft clips remain in stored sequences; hard-clipped bases are unavailable.
- Primary sequence bases: A/C/G/T/N and other observed symbol counts, total
  bases, GC fraction and N fraction, each fraction using all stored sequence
  bases as the denominator. Base composition is measured in alignment orientation.
- Available primary base qualities: histogram, total quality-bearing bases,
  arithmetic mean Phred score, Q20 and Q30 fractions. Missing qualities are
  counted separately and excluded from quality denominators.
- Complete header/read groups, per-contig reference MD5 results, input hashes,
  package versions, UTC completion time and QC elapsed seconds. Elapsed QC time
  includes integrity/reference verification; it excludes download and later
  provenance hashing.

These are acquisition and descriptive alignment QC metrics. They do not establish
zero-shot mapping accuracy, sample contamination, biological identity, exome
capture coverage or on-target fraction. Capture coverage needs the appropriate
capture BED and separate study rules. No trimming, filtering or duplicate removal
is applied during QC. A readiness pass does not choose experiment inclusion rules.

## Direct read streaming

`SequenceReader(cram_path, reference=decoding_fasta)` streams primary mapped and
unmapped reads, excludes secondary/supplementary records and restores original
read orientation. Read IDs retain mate suffixes. Its optional `alignment`
metadata retains read group, flags, contig, reference start/end, CIGAR and MAPQ
for downstream truth-assignment code; it is not automatically a gene truth label.
Current mapping CSV output does not export that metadata. When chunking a read,
this metadata still describes the source read; its CIGAR is not rewritten.

The `index`, `map` and `benchmark` commands accept `--reference` for CRAM input.
No global name sorting is needed when mates are embedded as individual reads.
The reader holds one record at a time; batches control embedding memory. Existing
query names plus mate suffixes are not guaranteed globally unique across read
groups; study sampling/truth tables must check uniqueness or include read group.
