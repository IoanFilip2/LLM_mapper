import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import pysam
from llm_mapper.preparation.downloads import Download, fetch
from llm_mapper.preparation.reference import extract_hgnc_reference, select_genes, decompress_fasta
from llm_mapper.preparation.samples import inspect_alignment, load_manifest, convert_cram
from llm_mapper.preparation.discovery import parse_igsr_index, https_url


def reference_fixture(tmp_path):
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">1\nAACCGGTTACGTRYSWKMBDHVN\n>X\nACGTACGT\n>Y\nTGCATGCA\n")
    pysam.faidx(str(fasta))
    hgnc = tmp_path / "hgnc.tsv"
    with hgnc.open("w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["hgnc_id", "symbol", "status", "locus_group", "ensembl_gene_id"])
        for ident, symbol, group, ensembl in [("HGNC:1", "CURRENT", "protein-coding gene", "ENSG1"),
                                             ("HGNC:2", "MINUS", "protein-coding gene", "ENSG2"),
                                             ("HGNC:3", "PAR", "protein-coding gene", "ENSG3"),
                                             ("HGNC:4", "MISSING", "protein-coding gene", "ENSG4"),
                                             ("HGNC:5", "NONCODING", "non-coding RNA", "ENSG5")]:
            writer.writerow([ident, symbol, "Approved", group, ensembl])
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(''.join(f'{chrom}\tEnsembl\tgene\t{start}\t{end}\t.\t{strand}\t.\tgene_id "{gene}"; gene_name "{symbol}"; gene_biotype "protein_coding";\n'
                           for chrom, start, end, strand, gene, symbol in [
                               ("1", 2, 8, "+", "ENSG1.2", "OLD"),
                               ("1", 9, 22, "-", "ENSG2", "MINUS"),
                               ("X", 1, 4, "+", "ENSG3", "PAR"),
                               ("Y", 1, 4, "+", "ENSG3_PAR_Y", "PAR"),
                               ("1", 1, 4, "+", "ENSG5", "NONCODING")]))
    return fasta, gtf, hgnc


def test_gene_spans_symbols_strand_and_audit(tmp_path):
    fasta, gtf, hgnc = reference_fixture(tmp_path)
    result = extract_hgnc_reference(fasta, gtf, hgnc, tmp_path / "out", True)
    assert result["genes_written"] == 3
    assert result["hgnc_approved_protein_coding"] == 4
    assert result["unresolved"][0]["symbol"] == "MISSING"
    with (tmp_path / "out" / "genes.csv").open() as handle:
        rows = {r["gene_symbol"]: r for r in csv.DictReader(handle)}
    assert rows["CURRENT"]["sequence"] == "ACCGGTT"
    assert rows["CURRENT"]["annotation_symbol"] == "OLD"
    assert rows["MINUS"]["sequence"] == "BDHVKMWSRYACGT"
    assert rows["PAR"]["chrom"] == "X"
    assert "1\t1\t8\tCURRENT\t0\t+" in (tmp_path / "out" / "genes.bed").read_text()
    assert (tmp_path / "out" / "genes.fa").read_text().startswith(">CURRENT ")
    with pytest.raises(FileExistsError):
        extract_hgnc_reference(fasta, gtf, hgnc, tmp_path / "out")


def test_ambiguous_genes_excluded_and_mismatch_fails(tmp_path):
    fasta, gtf, hgnc = reference_fixture(tmp_path)
    with gtf.open("a") as handle:
        handle.write('1\tEnsembl\tgene\t10\t15\t.\t+\t.\tgene_id "ENSG1"; gene_biotype "protein_coding";\n')
    selected, unresolved, _, _ = select_genes(gtf, hgnc)
    assert "CURRENT" not in {r["gene_symbol"] for r in selected}
    assert next(r for r in unresolved if r["symbol"] == "CURRENT")["reason"] == "multiple_loci"
    gtf.write_text(gtf.read_text().replace("\t22\t", "\t200\t"))
    with pytest.raises(ValueError, match="outside"):
        extract_hgnc_reference(fasta, gtf, hgnc, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_hgnc_link_takes_precedence_over_duplicate_symbol(tmp_path):
    _, gtf, hgnc = reference_fixture(tmp_path)
    with gtf.open("a") as handle:
        handle.write('1\tEnsembl\tgene\t10\t15\t.\t+\t.\tgene_id "ENSG_OTHER"; gene_name "CURRENT"; gene_biotype "protein_coding";\n')
    selected, _, excluded, _ = select_genes(gtf, hgnc)
    assert next(g for g in selected if g["gene_symbol"] == "CURRENT")["gene_id"] == "ENSG1"
    assert any(e["reason"] == "symbol_match_superseded_by_HGNC_linked_locus" for e in excluded)


def test_explicit_grch37_annotation_rejected(tmp_path):
    fasta, gtf, hgnc = reference_fixture(tmp_path)
    gtf.write_text("#!genome-build GRCh37\n" + gtf.read_text())
    with pytest.raises(ValueError, match="GRCh37"):
        extract_hgnc_reference(fasta, gtf, hgnc, tmp_path / "out")


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status, self.headers = status, headers or {"Content-Length": str(len(content)), "ETag": '"version"'}


def test_download_receipts_checksums_and_cache(tmp_path):
    content = b"real bytes"
    spec = Download("https://example.org/file", "file", md5=hashlib.md5(content).hexdigest())
    with patch("urllib.request.urlopen", return_value=Response(content)) as request:
        path = fetch(spec, tmp_path)
    assert path.read_bytes() == content
    with patch("urllib.request.urlopen") as request:
        assert fetch(spec, tmp_path) == path
        request.assert_not_called()
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        fetch(spec, tmp_path)


def test_download_resume_and_ignored_range(tmp_path):
    spec = Download("https://example.org/file", "file")
    (tmp_path / "file.part").write_bytes(b"abc")
    (tmp_path / "file.part.json").write_text(json.dumps({"url": spec.url, "etag": '"v1"'}))
    response = Response(b"def", 206, {"Content-Range": "bytes 3-5/6", "ETag": '"v1"'})
    with patch("urllib.request.urlopen", return_value=response) as request:
        assert fetch(spec, tmp_path).read_bytes() == b"abcdef"
        assert request.call_args.args[0].headers["Range"] == "bytes=3-"
    other = tmp_path / "other"
    other.mkdir()
    (other / "file.part").write_bytes(b"wrong")
    (other / "file.part.json").write_text(json.dumps({"url": spec.url, "etag": '"v1"'}))
    with patch("urllib.request.urlopen", return_value=Response(b"new content")):
        assert fetch(spec, other).read_bytes() == b"new content"


def test_bad_download_never_published(tmp_path):
    spec = Download("https://example.org/file", "file", md5="0" * 32)
    with patch("urllib.request.urlopen", return_value=Response(b"bad")), pytest.raises(ValueError, match="md5"):
        fetch(spec, tmp_path)
    assert not (tmp_path / "file").exists()
    with pytest.raises(ValueError):
        Download("https://example.org/file", "../escape")


def test_interrupted_download_automatically_resumes(tmp_path):
    spec = Download("https://example.org/file", "file")
    first = Response(b"abc", headers={"Content-Length": "6", "ETag": '"v1"'})
    second = Response(b"def", 206, {"Content-Range": "bytes 3-5/6", "ETag": '"v1"'})
    with patch("urllib.request.urlopen", side_effect=[first, second]), patch("time.sleep"):
        assert fetch(spec, tmp_path).read_bytes() == b"abcdef"


def test_layout_is_repeatable_and_separates_wgs(tmp_path):
    from llm_mapper.preparation.layout import initialize
    paths = initialize(tmp_path)
    assert {"reference", "catalog", "wes_100bp_1000genomes", "wes_250bp_public", "wgs_250bp_comparison"} == set(paths)
    readme = Path(paths["wgs_250bp_comparison"]) / "README.md"
    readme.write_text("user notes")
    initialize(tmp_path)
    assert readme.read_text() == "user notes"


def test_decompression_and_fasta_index(tmp_path):
    source = tmp_path / "ref.fa.gz"
    with gzip.open(source, "wb") as handle:
        handle.write(b">1\nACGT\n")
    dest = decompress_fasta(source, tmp_path / "ref.fa")
    assert Path(str(dest) + ".fai").exists()
    assert decompress_fasta(source, dest) == dest
    dest.write_text(">1\nTTTT\n")
    with pytest.raises(ValueError):
        decompress_fasta(source, dest)


def make_bam(path, lengths):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chr1", "LN": 248956422}, {"SN": "chr2", "LN": 242193529}, {"SN": "chrX", "LN": 156040895}],
              "RG": [{"ID": "run", "SM": "sample"}]}
    with pysam.AlignmentFile(path, "wb", header=header) as handle:
        for i, length in enumerate(lengths):
            read = pysam.AlignedSegment()
            read.query_name, read.query_sequence, read.flag = str(i), "A" * length, 4
            read.set_tag("RG", "run")
            handle.write(read)


@pytest.mark.parametrize("length", [100, 250])
def test_full_qc_and_prefix_not_verified(tmp_path, length):
    bam = tmp_path / "reads.bam"
    make_bam(bam, [length] * 10)
    qc = inspect_alignment(bam, length)
    assert qc["eligible_length_cohort"]
    assert qc["grch38_primary_lengths_match"]
    assert qc["length_histogram"] == {length: 10}
    assert not inspect_alignment(bam, length, limit=3)["eligible_length_cohort"]
    assert not inspect_alignment(bam, 150)["eligible_length_cohort"]


def test_mixed_read_lengths_are_not_trimmed(tmp_path):
    bam = tmp_path / "mixed.bam"
    make_bam(bam, [76, 100, 101, 150, 250])
    qc = inspect_alignment(bam, 100)
    assert not qc["eligible_length_cohort"]
    assert qc["exact_length_fraction"] == 0.2
    assert qc["length_histogram"] == {76: 1, 100: 1, 101: 1, 150: 1, 250: 1}


def test_manifest_strategy_guard_and_archived_ftp_index(tmp_path):
    row = {"sample_id": "S", "library_strategy": "WGS", "expected_read_length": 250,
           "format": "bam", "status": "selected", "evidence": ["official"]}
    row["alignment"] = {"url": "https://example.org/file.bam", "filename": "file.bam"}
    row["index"] = {"url": "https://example.org/file.bam.bai", "filename": "file.bam.bai"}
    path = tmp_path / "manifest.json"
    value = {"schema_version": 1, "cohort": "wes_250bp_public", "samples": [row]}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="strategy"):
        load_manifest(path)
    value["cohort"] = "wgs_250bp_comparison"
    path.write_text(json.dumps(value))
    assert load_manifest(path)["samples"][0]["sample_id"] == "S"
    assert https_url("ftp:/ftp.1000genomes.ebi.ac.uk/file") == "https://ftp.1000genomes.ebi.ac.uk/file"


def test_lossless_cram_conversion(tmp_path):
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">1\n" + "A" * 300 + "\n")
    pysam.faidx(str(fasta))
    source = tmp_path / "source.cram"
    header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "1", "LN": 300}]}
    with pysam.AlignmentFile(source, "wc", header=header, reference_filename=str(fasta)) as handle:
        read = pysam.AlignedSegment()
        read.query_name, read.query_sequence, read.flag = "r", "A" * 100, 0
        read.reference_id, read.reference_start, read.cigarstring = 0, 20, "100M"
        handle.write(read)
    bam = convert_cram(source, fasta, tmp_path / "out.bam")
    with pysam.AlignmentFile(bam, "rb") as handle:
        read = next(handle.fetch())
        assert (read.query_name, read.query_length, read.reference_start) == ("r", 100, 20)
    assert convert_cram(source, fasta, bam) == bam


def test_reviewed_cohort_scope_and_probe_evidence():
    root = Path(__file__).resolve().parents[1] / "configs"
    wes = load_manifest(root / "wes_100bp_1000genomes.json")
    wgs = load_manifest(root / "wgs_250bp_comparison.json")
    assert [s["sample_id"] for s in wes["samples"]] == ["NA19239", "NA20314"]
    assert [s["sample_id"] for s in wgs["samples"]] == ["HG00419"]
    for manifest in (wes, wgs):
        for sample in manifest["samples"]:
            assert sample["alignment"]["size"] > 0
            for evidence in sample["evidence"]:
                assert evidence["ena_metadata"]["library_strategy"] == sample["library_strategy"]
                assert len(evidence["fastq_prefix_probes"]) == 2
                assert all(p["matches_expected_length"] for p in evidence["fastq_prefix_probes"])


def test_direct_cram_qc_and_reader(tmp_path):
    from llm_mapper.sequences import SequenceReader
    fasta = tmp_path / 'ref.fa'
    fasta.write_text('>1\n' + 'A' * 300 + '\n')
    pysam.faidx(str(fasta))
    cram = tmp_path / 'reads.cram'
    header = {'HD': {'SO': 'coordinate'}, 'SQ': [{'SN': '1', 'LN': 300}],
              'RG': [{'ID': 'run', 'SM': 'sample'}]}
    with pysam.AlignmentFile(cram, 'wc', header=header, reference_filename=str(fasta)) as out:
        for i, flag in enumerate([81, 1024, 512, 256, 2048]):
            r = pysam.AlignedSegment()
            r.query_name = str(i)
            r.query_sequence = 'A' * 99 + 'N'
            r.query_qualities = pysam.qualitystring_to_array('?' * 100)
            r.flag = flag
            r.reference_id, r.reference_start = 0, i * 10
            r.cigarstring = '5S95M'
            r.mapping_quality = 42
            r.set_tag('RG', 'run')
            out.write(r)
    pysam.index(str(cram))
    qc = inspect_alignment(cram, 100, fasta)
    assert qc['full_decode_completed'] and qc['quickcheck_passed']
    assert qc['reference_verification']['all_header_md5_present_and_match']
    assert qc['total_records'] == 5 and qc['primary_records'] == 3
    assert qc['secondary_records'] == qc['supplementary_records'] == 1
    assert qc['primary_duplicates'] == qc['primary_qcfail'] == 1
    assert qc['base_counts'] == {'A': 297, 'N': 3}
    assert qc['quality_bases'] == 300 and qc['q30_fraction'] == 1
    assert qc['clipping']['soft_clipped_bases'] == 15
    reads = list(SequenceReader(cram, reference=fasta))
    assert len(reads) == 3 and reads[0].sequence == 'N' + 'T' * 99
    assert reads[0].id == '0/1'
    assert reads[0].alignment == {'read_group': 'run', 'flag': 81, 'contig': '1',
                                  'reference_start': 0, 'reference_end': 95,
                                  'cigar': '5S95M', 'mapping_quality': 42}
    assert not inspect_alignment(cram, 100, fasta, limit=10)['eligible_length_cohort']
    with pytest.raises(ValueError, match='reference'):
        list(SequenceReader(cram))
    wrong = tmp_path / 'wrong.fa'
    wrong.write_text('>1\n' + 'C' * 300 + '\n')
    with pytest.raises(ValueError, match='reference'):
        inspect_alignment(cram, 100, wrong)


def test_truncated_alignment_cannot_pass_qc(tmp_path):
    bam = tmp_path / 'truncated.bam'
    make_bam(bam, [100])
    bam.write_bytes(bam.read_bytes()[:-28])
    with pytest.raises(pysam.SamtoolsError):
        inspect_alignment(bam, 100)


def test_archive_comparison_uses_all_alignment_records(tmp_path):
    from llm_mapper.preparation.qc import compare_archive_statistics
    bas = tmp_path / 'sample.bas'
    bas.write_text('readgroup\t#_total_reads\na\t7\nb\t3\n')
    assert compare_archive_statistics(bas, {'total_records': 10, 'primary_records': 8})['total_matches']
    assert not compare_archive_statistics(bas, {'total_records': 8})['total_matches']
