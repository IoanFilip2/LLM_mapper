# Project status — 2026-09-27

Subsequent workflow update (2026-09-28): direct CRAM processing replaces mandatory
BAM conversion. See [NA19239 verification](na19239_verification.md) for the current
workflow and [completed QC report](NA19239_qc.md): 104,839,877 records decoded,
93,288,902 primary sequence-bearing records, all 100 bp, and all 3,366 reference
MD5s matched. No BAM intermediate was created. The remaining text records the
earlier preparation milestone.

The reference has been downloaded and processed. The active sequencing cohort
is now **two 100-bp WES samples and one separately labeled 250-bp WGS comparison
sample**. NA19239 source acquisition is complete and checksums passed; BAM conversion and
full-file QC are pending.
NA20314 is reserved for a later reproducibility study, and HG00419 for later
250-bp evaluation.

Development is on `feature/apple-silicon-v2`. The legacy `main` baseline remains
at `959b8df3bcd06835a6a5fb9f61d5a1cf8184133b`. This report describes workspace
progress; it does not imply that the changes have been committed or released.

## Completed implementation

- A common `SequenceEmbedder` interface, explicit CPU/MPS/CUDA selection,
  transformer adapters and an isolated experimental Evo2 adapter.
- Streaming FASTA, FASTQ, BAM and sequence-CSV input, configurable chunking and
  batching, preserving arbitrary read lengths including 100, 150 and 250 bp.
- A disk-sharded exact cosine index and configuration checks between reference
  and read embeddings, with index, mapping and benchmark CLI commands.
- Modular reference preparation, public-data discovery, resumable downloads,
  checksum receipts, CRAM-to-BAM conversion and full-file cohort QC.

GB.DNA-300M inference has run successfully on Apple Silicon MPS at all three
read lengths. The synthetic CPU/MPS smoke benchmark measured approximately
2.81–3.31× MPS throughput improvement. These results establish execution, not
biological mapping specificity. Memory figures are sampled process/device
allocations, not total system unified-memory usage; overlapping figures must
not be added. See [benchmark methods and results](../benchmarks/README.md).

DNABERT biological evaluation and a validated native Evo2 Apple Silicon loader
remain future work. The generic interface alone does not establish compatibility
with every checkpoint or remove Evo2's kernel requirements.

## Completed reference processing

The matched Ensembl release 110 GRCh38 primary-assembly FASTA and GTF were
downloaded with an HGNC complete-set snapshot last modified 2026-09-25.
The [source lock](../configs/grch38_ensembl110_hgnc_2026-09-25.json) records exact
hashes; retain the downloaded HGNC snapshot because its public URL is mutable.

Extraction produced **19,217 unique approved HGNC protein-coding gene symbols**
from 19,299 approved protein-coding records. The remaining **82 unresolved
records** are audited in the [reference report](reference_preparation_report.json).
No sequences were fabricated to reach a nominal 20,000 genes.

The sequences contain **full genomic gene spans, including introns**, with
minus-strand genes reverse-complemented. They total **1,318,920,029 bases** across
gene spans; overlapping genes can contribute overlapping genomic bases.
Selection uses primary chromosomes 1–22, X, Y and MT, prioritizes HGNC/Ensembl
ID links, and resolves matching X/Y copies to X. Metadata coordinates are
1-based inclusive; BED coordinates are 0-based, end-exclusive.

Active local outputs:

```text
data/reference/GRCh38-ensembl-110/
  sources/                  # Matched inputs, FASTA index and download receipts
  hgnc_genes/
    genes.fa                # HGNC-symbol FASTA, full genomic sequences
    genes.csv               # Metadata and sequence column
    genes.bed               # Gene spans, not an exome capture design
    report.json             # Counts, exclusions, unresolved loci and hashes
```

The earlier `hgnc_genes_initial_audit/` is retained for traceability. Downstream
work should use `hgnc_genes/`. A full biological reference embedding index has
not yet been built.

## Reduced sequencing cohort

| Cohort | Sample | Population | Native length per mate | Source CRAM size |
|---|---|---|---|---:|
| WES | NA19239 | YRI | 100 bp | 2.66 GB |
| WES | NA20314 | ASW | 100 bp | 3.10 GB |
| WGS comparison | HG00419 | CHS | 250 bp | 40.05 GB |

NA19239 and NA20314 have the smallest source CRAMs among the five previously
reviewed WES candidates. This selection reduces resource requirements; it is
not a claim of population representativeness. HG00238, HG00240 and HG01968 are
no longer active. Any existing metadata-only directories for those candidates
are retained; they do not control cohort membership or trigger downloads.

The active manifests are [two WES samples](../configs/wes_100bp_1000genomes.json)
and [one WGS comparison](../configs/wgs_250bp_comparison.json). The original ENA
discovery query is retained as provenance, while each manifest's `samples` list
defines the actual acquisition scope.

Selection combines official IGSR GRCh38 alignment metadata, BAS statistics,
ENA WXS/WGS classifications and 1,000 raw FASTQ records from each mate of every
contributing run. All **8,000 inspected reads** for the active WES pair were
100 bp; all **4,000 inspected WGS reads** were 250 bp. These prefix probes do
not replace full-file QC. No verified native 250-bp WES source is selected;
the [250-bp WES manifest](../configs/wes_250bp_public.json) remains pending.

Source CRAMs total **5,761,622,202 bytes for WES** and **40,045,057,118 bytes for
WGS**, or **45.81 GB combined** in decimal units. Reducing WES from five to two
saves approximately **19.19 GB** of source CRAM storage, plus associated BAM
storage and processing. These figures exclude references, indexes, temporary
files and expanded BAM derivatives. The WGS file remains the largest input.

Fewer samples reduce disk use and total processing time. Peak memory is governed
mainly by the model, sequence/chunk length, batch size and concurrent work.
Streaming and sequential sample processing keep the entire cohort out of RAM;
reducing the sample count alone does not guarantee a lower peak.

## Validation and remaining work

The preceding full test run recorded **38 passed, 1 skipped**; the skipped test
requires an external legacy gene CSV. The cohort regression test now requires
exactly NA19239 and NA20314 for WES and HG00419 for WGS.
After the cohort reduction, all **16 preparation tests passed**.

Full sequencing source acquisition completed for **NA19239 only** using
`--sample NA19239 --download-only`: the 2,657,487,300-byte CRAM, CRAI index and
BAS statistics passed downloader verification. Source files and receipts are in
`data/wes_100bp_1000genomes/NA19239/source/`; hashes and retrieval timestamps are
also recorded in the [download report](na19239_download_report.json).
NA20314 and HG00419 remain deferred. The exact IGSR GRCh38
full-analysis-set plus decoy/HLA reference required to decode those CRAMs is also
pending. It is separate from the Ensembl primary assembly used for gene extraction.
No converted sample BAM or full-file QC result is claimed here.

For the next preparation step on NA19239, run from the repository root. This
verifies cached sources, downloads the decoding reference, converts to BAM and
runs full-file QC:

```sh
.venv/bin/python -m llm_mapper.preparation samples configs/wes_100bp_1000genomes.json --sample NA19239
```

For the deferred cohorts, when ready:

```sh
.venv/bin/python -m llm_mapper.preparation samples configs/wes_100bp_1000genomes.json --sample NA20314
.venv/bin/python -m llm_mapper.preparation samples configs/wgs_250bp_comparison.json
```

Add `--sample NA19239` to the WES command to process just the first sample, or
`--download-only` to defer conversion and QC. The normal workflow downloads,
checksums, decodes, indexes and scans all primary sequence-bearing records.
Read-length eligibility requires at least 95% at the expected native length,
plus sample identity and assembly checks. Prefix-only QC cannot mark a sample
ready. Conversion does not trim reads or remove duplicates.

After acquisition, the mapping study still needs explicit truth assignment,
overlap/strand handling, sampling and evaluation rules, followed by matched
reference/read embedding configuration and biological accuracy, throughput and
memory measurements. See [the preparation guide](data_preparation.md) for
reproduction commands, source links and QC details.
