"""Official archive discovery; candidates never become verified cohorts implicitly."""
import csv
from dataclasses import asdict
import gzip
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
from .downloads import Download, fetch, file_hash, write_json

IGSR = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/"
DECODING_REFERENCE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa"


def https_url(value):
    if value.startswith("ftp:/"):
        return "https://" + value.split("ftp:", 1)[1].lstrip("/")
    if value.startswith(("http://", "https://")):
        return value.replace("http://", "https://", 1)
    return "https://" + value


def parse_igsr_index(path):
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 6:
                raise ValueError("Unexpected IGSR alignment index schema")
            downloads = [Download(https_url(fields[i]), fields[i].split("/")[-1], md5=fields[i+1]) for i in (0, 2, 4)]
            components = downloads[0].url.split("/")
            yield {"sample_id": components[-3], "population": components[-4],
                   "alignment": asdict(downloads[0]), "index": asdict(downloads[1]),
                   "statistics": asdict(downloads[2])}


def discover_igsr(samples, strategy, expected_length, output, cache="data/catalog"):
    if strategy not in {"WXS", "WGS"}:
        raise ValueError("Strategy must be WXS or WGS")
    source_name = "exome" if strategy == "WXS" else "high_coverage"
    filename = f"1000genomes.{source_name}.GRCh38DH.alignment.index"
    source = fetch(Download(IGSR + filename, filename), cache)
    rows = {r["sample_id"]: r for r in parse_igsr_index(source)}
    selected = []
    for sample in samples:
        row = rows[sample]
        bas = fetch(Download(**row["statistics"]), cache)
        with bas.open() as handle:
            stats = list(csv.DictReader(handle, delimiter="\t"))
        evidence = []
        for stat in stats:
            reads = int(stat["#_total_reads"])
            mean = int(stat["#_total_bases"]) / reads if reads else 0
            evidence.append({"run_accession": stat["readgroup"], "mean_aligned_query_length": mean,
                             "total_reads": reads, "source": row["statistics"]["url"]})
        # Small reductions can reflect hard clipping. This is only a candidate
        # screen: actual per-mate FASTQ probes and full alignment QC are separate.
        if not evidence or any(not expected_length * 0.97 <= e["mean_aligned_query_length"] <= expected_length + 0.1 for e in evidence):
            raise ValueError(f"{sample} has mixed/incompatible library lengths; review its BAS statistics")
        with urllib.request.urlopen(urllib.request.Request(row["alignment"]["url"], method="HEAD"), timeout=60) as response:
            if response.headers.get("Content-Length"):
                row["alignment"]["size"] = int(response.headers["Content-Length"])
        selected.append({**row, "library_strategy": strategy, "expected_read_length": expected_length,
                         "format": "cram", "assembly": "GRCh38DH", "status": "selected",
                         "evidence": evidence,
                         "validation_note": "BAS mean length is candidate evidence only; full-file native query length QC is required after download"})
    cohort = ("wes_100bp_1000genomes" if expected_length == 100 else "wes_250bp_public") if strategy == "WXS" else "wgs_250bp_comparison"
    manifest = {"schema_version": 1, "cohort": cohort, "source_index_url": IGSR + filename,
                "source_index_sha256": file_hash(source), "samples": selected,
                "decoding_reference": asdict(Download(DECODING_REFERENCE, DECODING_REFERENCE.split("/")[-1]))}
    write_json(output, manifest)
    return manifest


def discover_ena(query, output):
    fields = ["run_accession", "experiment_accession", "study_accession", "sample_accession", "sample_alias",
              "study_title", "experiment_title", "instrument_model", "library_strategy", "library_layout",
              "read_count", "base_count", "fastq_ftp", "fastq_md5", "submitted_ftp", "submitted_md5"]
    url = "https://www.ebi.ac.uk/ena/portal/api/search?" + urllib.parse.urlencode(
        {"result": "read_run", "query": query, "fields": ",".join(fields), "format": "tsv", "limit": 0})
    with urllib.request.urlopen(url, timeout=120) as response:
        table = response.read().decode()
    records = list(csv.DictReader(io.StringIO(table), delimiter="\t"))
    result = {"status": "candidates_only", "query_url": url, "records": records,
              "note": "ENA read_count may count spots. base_count/read_count is not proof of per-mate read length. "
                      "Inspect experiment methods and actual FASTQ/BAM; WXS metadata can include targeted panels."}
    write_json(output, result)
    return result


def probe_fastq(url, limit=1000):
    """Remote prefix probe, not a full-file checksum or representative QC scan."""
    from collections import Counter
    if limit < 1:
        raise ValueError("Probe limit must be positive")
    counts = Counter()
    with urllib.request.urlopen(https_url(url), timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as compressed, io.TextIOWrapper(compressed) as handle:
            for _ in range(limit):
                header = handle.readline()
                if not header:
                    break
                sequence, separator, quality = handle.readline().rstrip(), handle.readline(), handle.readline().rstrip()
                if not header.startswith("@") or not separator.startswith("+") or len(sequence) != len(quality):
                    raise ValueError("Malformed/non-four-line FASTQ during remote probe")
                counts[len(sequence)] += 1
    return {"url": https_url(url), "scope": "prefix_only", "reads_checked": sum(counts.values()),
            "length_histogram": dict(sorted(counts.items()))}


def verify_manifest_fastqs(manifest_path, limit=1000):
    """Add direct per-mate raw read evidence, retaining selected (not full-QC) status."""
    from concurrent.futures import ThreadPoolExecutor
    from .samples import load_manifest
    manifest = load_manifest(manifest_path)
    runs = sorted({e["run_accession"] for s in manifest["samples"] for e in s["evidence"]})
    query = " OR ".join(f'run_accession="{run}"' for run in runs)
    fields = "run_accession,study_accession,sample_accession,library_strategy,library_layout,fastq_ftp,fastq_md5,read_count,base_count"
    url = "https://www.ebi.ac.uk/ena/portal/api/search?" + urllib.parse.urlencode(
        dict(result="read_run", query=query, fields=fields, format="tsv", limit=0))
    with urllib.request.urlopen(url, timeout=120) as response:
        by_run = {r["run_accession"]: r for r in csv.DictReader(io.StringIO(response.read().decode()), delimiter="\t")}
    tasks = []
    for sample in manifest["samples"]:
        for evidence in sample["evidence"]:
            row = by_run[evidence["run_accession"]]
            if row["library_strategy"] != sample["library_strategy"]:
                raise ValueError(f"ENA library strategy disagrees for {evidence['run_accession']}")
            evidence["ena_metadata"] = row
            evidence["fastq_prefix_probes"] = []
            mates = [u for u in row["fastq_ftp"].split(";") if u.endswith(("_1.fastq.gz", "_2.fastq.gz"))]
            if len(mates) != 2:
                raise ValueError(f"Two mate files required for {evidence['run_accession']}")
            tasks.extend((evidence, u, sample["expected_read_length"]) for u in mates)
    def probe(task):
        evidence, url, length = task
        result = probe_fastq(url, limit)
        result["matches_expected_length"] = result["reads_checked"] > 0 and set(result["length_histogram"]) == {length}
        return evidence, result
    with ThreadPoolExecutor(max_workers=4) as pool:
        for evidence, result in pool.map(probe, tasks):
            evidence["fastq_prefix_probes"].append(result)
    manifest["ena_query_url"] = url
    manifest["native_length_prefixes_match"] = all(
        p["matches_expected_length"] for s in manifest["samples"] for e in s["evidence"]
        for p in e["fastq_prefix_probes"]
    )
    write_json(manifest_path, manifest)
    if not manifest["native_length_prefixes_match"]:
        raise ValueError("Raw read-length probe disagrees with the requested cohort; evidence saved for review")
    return manifest
