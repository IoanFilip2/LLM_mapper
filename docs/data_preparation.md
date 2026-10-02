# Reference and sequencing preparation

This stage prepares source data and records provenance. It does not choose the
mapping study's truth-assignment, overlap, strand, sampling, or evaluation rules.
The fixed selection is **two 100-bp WES samples and one 250-bp WGS test sample**.
WGS is the explicitly approved comparison cohort; it is never labeled WES.

## Local reference already prepared

The reference uses the matched **Ensembl release 110 GRCh38 primary-assembly
FASTA and GTF** from the existing download script, with an HGNC complete-set
snapshot last modified 2026-09-25. Release 110 is from 2023, not 2024.

The run produced **19,217 unique approved HGNC protein-coding symbols**, from
19,299 approved protein-coding HGNC records, containing **1,318,920,029 bases**
across gene spans. The remaining **82** HGNC records have no matching supported
primary-chromosome protein-coding locus in this annotation. This includes
complex loci/alternate-haplotype genes and annotation/catalog differences; no
sequences are invented to reach exactly 20,000. The complete unresolved list and
input/output hashes are in [the preparation report](reference_preparation_report.json).

Selection uses primary chromosomes 1–22, X, Y, and MT, GTF protein-coding gene
features, and approved HGNC protein-coding membership. HGNC/Ensembl ID links
outrank symbol-only matches. Same-ID X/Y duplicates retain X; unresolved genuine
multiple-locus conflicts are audited rather than arbitrarily combined. The
HGNC-linked X copy is selected for PAR genes when Y has only a symbol match.
Symbols are current approved HGNC symbols; annotation symbols are also retained.

Sequences are **full genomic gene spans, including introns**, not CDS, merged
exons, or transcript sequences. Minus-strand spans are reverse-complemented into
gene orientation. Metadata coordinates remain 1-based inclusive; BED is 0-based,
end-exclusive. No flanking regions are added. Genomic offsets for minus-strand
sequence chunks must be reversed when assigning coordinate-based truth later.

Prepared files:

```text
data/reference/GRCh38-ensembl-110/
  sources/                  # FASTA, GTF, HGNC snapshot, indexes, checksum receipts
  hgnc_genes/
    genes.fa                # Full sequences; FASTA ID is the approved HGNC symbol
    genes.csv               # IDs, symbols, coordinates, strand, match method, sequence
    genes.bed               # Full gene spans (not an exome capture BED)
    report.json             # Counts, unresolved/excluded loci, hashes
```

The first extraction audit is retained in `hgnc_genes_initial_audit/`; use
`hgnc_genes/` for downstream inputs. The CSV sequence column is optional in new
runs (`--sequence-csv`); FASTA always contains full sequences. The existing
`SequenceReader` accepts the FASTA directly or a CSV that includes `sequence`.
Each gene is fetched through the FASTA index, so the whole genome is not loaded
into Python memory.

## Selected public inputs

Sources are the official IGSR GRCh38 alignment indexes. They provide CRAM and
CRAI files with MD5 checksums. Direct decoding uses the **exact IGSR
GRCh38 full analysis set plus decoy/HLA reference**, kept separately from the
Ensembl reference used for gene extraction. No GRCh37 coordinates are relabeled
as GRCh38, and changing `chr1` to `1` is never treated as assembly conversion.

| Cohort | Sample | Population | Native library read length |
|---|---|---|---|
| WES | NA19239 | YRI | 100 bp per mate |
| WES | NA20314 | ASW | 100 bp per mate |
| WGS comparison only | HG00419 | CHS | 250 bp per mate |

Selection evidence combines IGSR's exome/high-coverage collection, read-group
BAS statistics, matching ENA WXS/WGS classifications, and direct inspection of
1,000 FASTQ records from **each mate of every contributing run**. All 8,000
inspected WES reads were 100 bp; all 4,000 inspected HG00419 WGS reads were 250 bp.
These are **prefix probes**, not claims that every alignment record has that
length. Hard-clipped alignments can be shorter. Full-file QC is mandatory after
acquisition. HG03702 was excluded because its ENA strategy disagreed with the
IGSR exome collection; it is not in the active two-sample manifest.

The reviewed manifests include URLs, upstream MD5s, raw FASTQ metadata and probe
results, alignment assembly, expected read lengths, and source sizes:

- [Two WES samples](../configs/wes_100bp_1000genomes.json)
- [One WGS comparison sample](../configs/wgs_250bp_comparison.json)
- [250-bp WES status](../configs/wes_250bp_public.json): no verified sources selected

The two WES CRAMs total approximately **5.76 GB**, and the single WGS CRAM is
approximately **40.05 GB**. BAM export is optional and requires additional
storage; the original CRAM is the canonical input for streaming and QC.
The complete **NA19239 CRAM, CRAI and BAS files have been downloaded and
verified**; see the [download report](na19239_download_report.json).

**Direct full-file QC passed on 2026-09-28**: 104,839,877 alignment records,
including 93,288,902 primary sequence-bearing reads, all 100 bp. See
[the NA19239 workflow](na19239_verification.md) and [QC report](NA19239_qc.md).
NA20314 is deferred for reproducibility, and HG00419 for later 250-bp evaluation.

The two WES samples were chosen as the smallest source CRAMs in the previously
reviewed five-sample pool. Together with WGS, source CRAM storage is approximately
**45.81 GB**, excluding references, indexes and BAM derivatives. Fewer samples
reduce storage and total processing; peak memory still depends on model size,
sequence lengths and batch size. Previously created metadata-only directories
for other candidates are inactive; the reviewed manifests control downloads.

Directory layout:

```text
data/reference/
data/catalog/
data/wes_100bp_1000genomes/<sample>/
  source/                   # Original CRAM, CRAI, BAS and download receipts
  <sample>.GRCh38.bam        # Optional --export-bam derivative only
  source_manifest.json
  qc.json
data/wes_250bp_public/       # Reserved; empty until actual WES sources qualify
data/wgs_250bp_comparison/HG00419/
```

## Commands

Install the preparation dependencies and create the directories:

```sh
source .venv/bin/activate
pip install -e '.[prepare,test]'
python -m llm_mapper.preparation init
```

Prepare or verify the already-downloaded reference:

```sh
python scripts/download_reference.py --sequence-csv \
  --source-manifest configs/grch38_ensembl110_hgnc_2026-09-25.json
```

The [reference lock](../configs/grch38_ensembl110_hgnc_2026-09-25.json) checks exact
source SHA-256 hashes. HGNC's live URL is mutable: preserve the local snapshot
and receipts. A future replacement at that URL will fail the lock rather than
silently change gene labels. Omitting `--source-manifest` explicitly uses the
current HGNC catalog on a fresh download and therefore may change the result.
Existing cached downloads remain pinned to their receipts. Changing annotation
release requires a new output directory; downstream indexes must then be rebuilt.

To extract from existing matched local files:

```sh
python -m llm_mapper.preparation extract \
  --fasta data/reference/GRCh38-ensembl-110/sources/GRCh38.primary_assembly.fa \
  --gtf data/reference/GRCh38-ensembl-110/sources/GRCh38.ensembl-110.gtf.gz \
  --hgnc data/reference/GRCh38-ensembl-110/sources/hgnc_complete_set.tsv \
  --output data/reference/custom_hgnc_genes --sequence-csv
```

Download and directly QC the selected WES sample:

```sh
python -m llm_mapper.preparation samples configs/wes_100bp_1000genomes.json --sample NA19239
```

Download and prepare **only HG00419** for the WGS comparison:

```sh
python -m llm_mapper.preparation samples configs/wgs_250bp_comparison.json
```

`--sample NA19239` restricts processing to the first WES sample; `--download-only` acquires
original files without QC. Add `--export-bam` only if a BAM derivative is needed. `python scripts/download_small_bam.py`
now delegates to the reviewed WES manifest; despite its historical name, it no
longer downloads a mislabeled chromosome-specific genome demo.

Downloads use `.part` files, HTTP Range/If-Range resume, bounded retry, expected
size and upstream MD5 checks, local SHA-256 receipts, and atomic publication.
Existing files are verified, not accepted solely because they exist. Interrupted
commands can be rerun. On checksum mismatch the command stops; inspect/quarantine
the invalid `.part` file before retrying. Concurrent processes targeting the same
output file are not supported.

Direct CRAM processing retains sequences, flags, and alignments from the source; it
applies no trimming, filtering, re-alignment, or duplicate removal. It does not
restore information already discarded by the original archive's processing.
QC scans every primary sequence-bearing record and records the native length
histogram, per-read-group counts, duplicates, unmapped reads, sample identity,
and GRCh38 chr1/chr2/chrX length checks. At least 95% must have the expected native
length to pass the cohort gate. A failure leaves source data and a report for
review, not a falsely ready cohort. The 95% gate is an acquisition check, not a
mapping-study inclusion rule. Assembly length checks supplement the official
source provenance; they are not a full genome-sequence equivalence test.

For explicit inspection of an existing BAM:

```sh
python -m llm_mapper.preparation inspect sample.bam \
  --expected-length 100 --output sample.qc.json
```

`--limit N` is a fast prefix inspection and cannot mark a cohort verified.

## Repeatable discovery

```sh
python scripts/discover_public_data.py igsr \
  --samples NA19239 NA20314 \
  --strategy WXS --length 100 --output two-wes.json
python scripts/discover_public_data.py verify-manifest two-wes.json

python scripts/discover_public_data.py igsr \
  --samples HG00419 --strategy WGS --length 250 --output one-wgs.json
python scripts/discover_public_data.py verify-manifest one-wgs.json

python scripts/discover_public_data.py ena --output data/catalog/wes_candidates.json
```

ENA discovery produces **candidates**, not approved downloads. `WXS` tags may
refer to targeted panels, and ENA read counts may count paired spots; neither
instrument name, insert size, nor `base_count/read_count` alone proves per-mate
read length. The reviewed 250-bp WES candidates included targeted panels and
shorter-read GIAB exomes; these were not substituted into the WES cohort.

Primary sources:

- [IGSR GRCh38 exome/high-coverage alignment announcement](https://www.internationalgenome.org/announcements/grch38-alignments-exome-and-high-coverage-1000-genomes-data-2015-12-16/)
- [Official IGSR indexes and alignment README](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/)
- [Ensembl release 110 GTF files](https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/)
- [HGNC complete-set download](https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt)
- [ENA portal API](https://www.ebi.ac.uk/ena/portal/api/)
