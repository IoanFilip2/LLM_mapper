"""Manifest-driven WES acquisition, direct CRAM QC, and optional BAM export."""
import json
from pathlib import Path
import re
from .downloads import Download, fetch, file_hash, write_json

GRCH38_LENGTHS = {"1": 248956422, "2": 242193529, "X": 156040895}


def canonical_contig(name):
    name = name.removeprefix("chr")
    return "MT" if name == "M" else name


from .qc import inspect_alignment, compare_archive_statistics


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported sample manifest version")
    cohorts = {"wes_100bp_1000genomes": ("WXS", 100),
               "wes_250bp_public": ("WXS", 250), "wgs_250bp_comparison": ("WGS", 250)}
    if manifest.get("cohort") not in cohorts:
        raise ValueError("Unknown cohort; WGS and WES must be explicitly separated")
    ids = set()
    for sample in manifest["samples"]:
        identifier = sample["sample_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) or identifier in {".", ".."} or identifier in ids:
            raise ValueError("Invalid or duplicate sample ID")
        ids.add(identifier)
        if (sample["library_strategy"], sample["expected_read_length"]) != cohorts[manifest["cohort"]]:
            raise ValueError("Library strategy/read length does not match the declared cohort")
        if sample["status"] not in {"selected", "verified"}:
            raise ValueError(f"Unreviewed candidate cannot be downloaded as a cohort: {identifier}")
        if not sample.get("evidence"):
            raise ValueError("Samples require library/read-length evidence")
        if sample["expected_read_length"] not in {100, 250}:
            raise ValueError("This cohort requires native 100-bp or 250-bp reads")
        if sample["format"] not in {"bam", "cram"}:
            raise ValueError("Prepared samples must be BAM or CRAM")
        for key in ("alignment", "index"):
            Download(**sample[key])
        if sample["format"] == "cram" and not manifest.get("decoding_reference"):
            raise ValueError("CRAM decoding requires its exact decoding reference")
    return manifest


def convert_cram(source, reference, destination, threads=2):
    import pysam
    source, reference, destination = Path(source), Path(reference), Path(destination)
    receipt = destination.with_name(destination.name + ".conversion.json")
    source_sha256, reference_sha256 = file_hash(source), file_hash(reference)
    if destination.exists():
        if not receipt.exists():
            raise FileExistsError(f"Untracked existing BAM: {destination}")
        previous = json.loads(receipt.read_text())
        if (previous["source_sha256"] != source_sha256 or previous["reference_sha256"] != reference_sha256
                or previous["bam_sha256"] != file_hash(destination)):
            raise ValueError("Existing converted BAM provenance/checksum mismatch")
    else:
        temporary = destination.with_name(destination.stem + ".part.bam")
        if not Path(str(reference) + ".fai").exists():
            pysam.faidx(str(reference))
        pysam.view("-@", str(threads), "-b", "-T", str(reference), "-o", str(temporary), str(source), catch_stdout=False)
        pysam.quickcheck(str(temporary))
        result = {"source_sha256": source_sha256, "reference_sha256": reference_sha256,
                  "bam_sha256": file_hash(temporary), "pysam_version": pysam.__version__,
                  "transformation": "CRAM decoding only; original alignments, sequences and flags retained"}
        write_json(receipt, result)
        temporary.replace(destination)
    if not Path(str(destination) + ".bai").exists():
        pysam.index("-@", str(threads), str(destination))
    return destination


def prepare_samples(manifest_path, root="data", selected=None, download_only=False, export_bam=False):
    import pysam
    manifest = load_manifest(manifest_path)
    root = Path(root)
    samples = [s for s in manifest["samples"] if selected is None or s["sample_id"] in selected]
    if selected and set(selected) != {s["sample_id"] for s in samples}:
        raise ValueError("Requested sample is absent from the manifest")
    if not samples:
        raise ValueError("No verified/selected samples in this manifest; discovery is still pending")
    decoding_reference = None
    if not download_only and any(s["format"] == "cram" for s in samples):
        decoding_reference = fetch(Download(**manifest["decoding_reference"]), root / "reference" / "GRCh38-IGSR-decoding")
    for sample in samples:
        cohort = manifest["cohort"]
        directory = root / cohort / sample["sample_id"]
        print(f"Preparing {sample['sample_id']} ({sample['format']}, native {sample['expected_read_length']} bp expected)", flush=True)
        alignment = fetch(Download(**sample["alignment"]), directory / "source")
        fetch(Download(**sample["index"]), directory / "source")
        if sample.get("statistics"):
            fetch(Download(**sample["statistics"]), directory / "source")
        write_json(directory / "source_manifest.json", sample)
        if download_only:
            continue
        if export_bam and sample["format"] == "cram":
            convert_cram(alignment, decoding_reference, directory / f"{sample['sample_id']}.GRCh38.bam")
        qc = inspect_alignment(alignment, sample["expected_read_length"], decoding_reference)
        if sample.get("statistics"):
            qc["archive_statistics_comparison"] = compare_archive_statistics(
                directory / "source" / sample["statistics"]["filename"], qc)
        qc["sample_id"] = sample["sample_id"]
        qc["library_strategy"] = sample["library_strategy"]
        qc["alignment_sha256"] = file_hash(alignment)
        qc["reference_sha256"] = file_hash(decoding_reference) if decoding_reference else None
        qc["source_manifest_sha256"] = file_hash(directory / "source_manifest.json")
        header_samples = {group.get("SM") for group in qc["read_groups"]}
        qc["sample_id_matches_header"] = header_samples == {sample["sample_id"]}
        qc["ready_for_downstream"] = (qc["eligible_length_cohort"] and qc["grch38_primary_lengths_match"]
                                      and qc["sample_id_matches_header"] and qc["full_decode_completed"]
                                      and qc.get("archive_statistics_comparison", {}).get("total_matches", True)
                                      and (qc["reference_verification"] is None or
                                           qc["reference_verification"]["all_header_md5_present_and_match"]))
        write_json(directory / "qc.json", qc)
        if not qc["ready_for_downstream"]:
            raise ValueError(f"{sample['sample_id']} failed length/assembly QC; see {directory / 'qc.json'}")
        print(f"Ready: {alignment} ({qc['primary_reads_with_sequence']} primary reads)", flush=True)
