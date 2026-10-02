"""HGNC-approved protein-coding gene spans from a matched GRCh38 FASTA/GTF."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
import shutil
import tempfile
from .downloads import Download, fetch, file_hash, write_json
from ..extract_gene_sequences import reverse_complement

ENSEMBL_RELEASE = 110
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
PRIMARY = {str(i) for i in range(1, 23)} | {"X", "Y", "MT"}


def reference_sources(release=ENSEMBL_RELEASE):
    root = f"https://ftp.ensembl.org/pub/release-{release}"
    return [Download(f"{root}/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
                     "GRCh38.primary_assembly.fa.gz"),
            Download(f"{root}/gtf/homo_sapiens/Homo_sapiens.GRCh38.{release}.gtf.gz",
                     f"GRCh38.ensembl-{release}.gtf.gz"),
            Download(HGNC_URL, "hgnc_complete_set.tsv")]


def decompress_fasta(source, destination):
    import pysam
    source, destination = Path(source), Path(destination)
    provenance = destination.with_name(destination.name + ".source.json")
    digest = file_hash(source)
    if destination.exists():
        if not provenance.exists():
            raise FileExistsError(f"Untracked FASTA: {destination}")
        previous = json.loads(provenance.read_text())
        if previous["source_sha256"] != digest or previous["sha256"] != file_hash(destination):
            raise ValueError("Decompressed FASTA checksum mismatch")
    else:
        part = destination.with_name(destination.name + ".part")
        with gzip.open(source, "rb") as input_, part.open("wb") as output:
            shutil.copyfileobj(input_, output, length=1024 * 1024)
        write_json(provenance, {"source_sha256": digest, "sha256": file_hash(part)})
        part.replace(destination)
    if not Path(str(destination) + ".fai").exists():
        pysam.faidx(str(destination))
    return destination


def gtf_records(path, feature="gene"):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Invalid GTF line {number}")
            if fields[2] not in ({feature} if isinstance(feature, str) else set(feature)):
                continue
            pairs = re.findall(r'(\w+)\s+"([^"]*)"', fields[8])
            attrs = dict(pairs)
            attrs["tags"] = [v for k, v in pairs if k == "tag"]
            start, end = int(fields[3]), int(fields[4])
            if start < 1 or end < start or fields[6] not in {"+", "-"}:
                raise ValueError(f"Invalid GTF coordinates on line {number}")
            yield {"chrom": fields[0], "start": start, "end": end,
                   "strand": fields[6], "feature": fields[2], "phase": fields[7], **attrs}


def check_gtf_build(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            if line.startswith("#!genome-build "):
                build = line.split(maxsplit=1)[1].strip()
                if not build.startswith("GRCh38"):
                    raise ValueError(f"Expected GRCh38 annotation, got {build}")
                return build
    return None


def approved_hgnc(path):
    with open(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"hgnc_id", "symbol", "status", "locus_group", "ensembl_gene_id"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("HGNC table is missing required columns")
        rows = [row for row in reader if row["status"] == "Approved"
                and row["locus_group"] == "protein-coding gene"]
    for field in ("hgnc_id", "symbol"):
        if len({r[field] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate approved HGNC {field}")
    return rows


def select_genes(gtf, hgnc):
    approved = approved_hgnc(hgnc)
    by_id = {r["hgnc_id"]: r for r in approved}
    by_symbol = {r["symbol"]: r for r in approved}
    by_ensembl = defaultdict(list)
    for row in approved:
        if row["ensembl_gene_id"]:
            by_ensembl[row["ensembl_gene_id"]].append(row)
    candidates, excluded = defaultdict(list), []
    for gene in gtf_records(gtf):
        chrom = gene["chrom"].removeprefix("chr")
        if chrom == "M":
            chrom = "MT"
        if chrom not in PRIMARY:
            continue
        if gene.get("gene_biotype", gene.get("gene_type")) != "protein_coding":
            continue
        ensembl = gene.get("gene_id", "").split(".")[0].removesuffix("_PAR_Y")
        matches = by_ensembl.get(ensembl, [])
        hgnc_row = by_id.get(gene.get("hgnc_id"))
        if hgnc_row:
            match, method = hgnc_row, "hgnc_id"
        elif len(matches) == 1:
            match, method = matches[0], "ensembl_gene_id"
        elif len(matches) > 1:
            excluded.append({"gene_id": ensembl, "reason": "ambiguous_hgnc_ensembl_id"})
            continue
        elif gene.get("gene_name") in by_symbol:
            match, method = by_symbol[gene["gene_name"]], "approved_symbol"
        else:
            excluded.append({"gene_id": ensembl, "reason": "no_approved_hgnc_match"})
            continue
        candidates[match["hgnc_id"]].append({
            "hgnc_id": match["hgnc_id"], "gene_symbol": match["symbol"],
            "gene_id": ensembl, "annotation_symbol": gene.get("gene_name", ""),
            "chrom": gene["chrom"], "start": gene["start"], "end": gene["end"],
            "strand": gene["strand"], "match_method": method, "assembly": "GRCh38",
            "sequence_length": gene["end"] - gene["start"] + 1,
        })
    selected, unresolved = [], []
    for row in approved:
        genes = candidates.get(row["hgnc_id"], [])
        linked = [g for g in genes if g["match_method"] in {"hgnc_id", "ensembl_gene_id"}]
        if len(genes) > 1 and len(linked) == 1:
            selected.append(linked[0])
            for gene in genes:
                if gene is not linked[0]:
                    excluded.append({"hgnc_id": row["hgnc_id"], "gene_id": gene["gene_id"],
                                     "chrom": gene["chrom"], "reason": "symbol_match_superseded_by_HGNC_linked_locus"})
            continue
        if len(genes) == 1:
            selected.append(genes[0])
        elif len(genes) == 2 and {g["chrom"].removeprefix("chr") for g in genes} == {"X", "Y"} and len({g["gene_id"] for g in genes}) == 1:
            selected.append(next(g for g in genes if g["chrom"].removeprefix("chr") == "X"))
            excluded.append({"hgnc_id": row["hgnc_id"], "reason": "PAR_Y_duplicate_X_retained"})
        else:
            unresolved.append({"hgnc_id": row["hgnc_id"], "symbol": row["symbol"],
                               "reason": "no_matching_primary_protein_coding_locus" if not genes else "multiple_loci",
                               "candidates": genes})
    return sorted(selected, key=lambda r: r["gene_symbol"]), unresolved, excluded, len(approved)


def extract_hgnc_reference(fasta, gtf, hgnc, destination, include_sequence_csv=False):
    """One gene per approved symbol; full genomic span oriented 5'→3'.

    BED coordinates are 0-based half-open. Metadata start/end retain GTF's
    1-based inclusive convention. Ambiguous/missing loci are explicitly audited.
    """
    import pysam
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    declared_build = check_gtf_build(gtf)
    genes, unresolved, excluded, n_approved = select_genes(gtf, hgnc)
    if not genes:
        raise ValueError("No HGNC-approved protein-coding genes matched")
    staging = Path(tempfile.mkdtemp(prefix=".genes-", dir=destination.parent))
    fields = list(genes[0])
    try:
        with pysam.FastaFile(str(fasta)) as genome, (staging / "genes.fa").open("w") as sequences, (staging / "genes.csv").open("w", newline="") as metadata, (staging / "genes.bed").open("w") as bed:
            writer = csv.DictWriter(metadata, fieldnames=fields + (["sequence"] if include_sequence_csv else []))
            writer.writeheader()
            for gene in genes:
                chrom = gene["chrom"]
                if chrom not in genome.references:
                    raise ValueError(f"GTF/FASTA contig mismatch: {chrom}; use matching releases")
                if gene["end"] > genome.get_reference_length(chrom):
                    raise ValueError(f"Gene outside reference: {gene['gene_symbol']}")
                seq = genome.fetch(chrom, gene["start"] - 1, gene["end"]).upper()
                if gene["strand"] == "-":
                    seq = reverse_complement(seq)
                if len(seq) != gene["sequence_length"]:
                    raise ValueError(f"Truncated gene: {gene['gene_symbol']}")
                sequences.write(f">{gene['gene_symbol']} hgnc_id={gene['hgnc_id']} gene_id={gene['gene_id']} {chrom}:{gene['start']}-{gene['end']} strand={gene['strand']}\n")
                for offset in range(0, len(seq), 80):
                    sequences.write(seq[offset:offset+80] + "\n")
                writer.writerow({**gene, **({"sequence": seq} if include_sequence_csv else {})})
                bed.write(f"{chrom}\t{gene['start']-1}\t{gene['end']}\t{gene['gene_symbol']}\t0\t{gene['strand']}\n")
        report = {"assembly": "GRCh38", "scope": "approved HGNC protein-coding genes on primary chromosomes",
                  "gtf_declared_build": declared_build,
                  "sequence_type": "full genomic gene span including introns; strand oriented",
                  "sequence_csv_included": include_sequence_csv,
                  "hgnc_approved_protein_coding": n_approved, "genes_written": len(genes),
                  "total_gene_bases": sum(g["sequence_length"] for g in genes),
                  "unresolved": unresolved, "excluded": excluded,
                  "inputs": {key: {"path": str(path), "sha256": file_hash(path)}
                             for key, path in {"fasta": fasta, "gtf": gtf, "hgnc": hgnc}.items()},
                  "outputs": {name: file_hash(staging / name) for name in ("genes.fa", "genes.csv", "genes.bed")}}
        write_json(staging / "report.json", report)
        staging.rename(destination)
        return report
    except BaseException:
        shutil.rmtree(staging)
        raise


def prepare_reference(root="data/reference", release=ENSEMBL_RELEASE, include_sequence_csv=False, source_manifest=None):
    root = Path(root) / f"GRCh38-ensembl-{release}"
    sources = reference_sources(release)
    if source_manifest:
        locked = json.loads(Path(source_manifest).read_text())
        if locked["assembly"] != "GRCh38" or locked["ensembl_release"] != release:
            raise ValueError("Reference manifest assembly/release mismatch")
        sources = [Download(**locked["sources"][role]) for role in ("fasta_gz", "gtf_gz", "hgnc")]
    paths = [fetch(source, root / "sources") for source in sources]
    fasta = decompress_fasta(paths[0], root / "sources" / "GRCh38.primary_assembly.fa")
    destination = root / "hgnc_genes"
    if destination.exists():
        report = json.loads((destination / "report.json").read_text())
        for key, path in {"fasta": fasta, "gtf": paths[1], "hgnc": paths[2]}.items():
            if file_hash(path) != report["inputs"][key]["sha256"]:
                raise ValueError("Existing gene reference has different inputs; use a new output directory")
        for name, digest in report["outputs"].items():
            if file_hash(destination / name) != digest:
                raise ValueError(f"Existing gene reference output has changed: {name}")
        if include_sequence_csv and not report.get("sequence_csv_included"):
            raise ValueError("Existing output lacks sequence CSV; use extract with a new output directory")
        return report
    return extract_hgnc_reference(fasta, paths[1], paths[2], destination, include_sequence_csv)
