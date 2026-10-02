"""Bounded-memory full alignment decoding and descriptive QC with pysam."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time


def verify_reference(header, reference):
    """Compare every alignment @SQ length and available M5 against local FASTA."""
    import pysam
    results = []
    with pysam.FastaFile(str(reference)) as fasta:
        names = set(fasta.references)
        for sq in header.get('SQ', []):
            name = sq['SN']
            row = {'contig': name, 'header_length': sq['LN'], 'header_md5': sq.get('M5')}
            row['present'] = name in names
            row['length_matches'] = name in names and fasta.get_reference_length(name) == sq['LN']
            if row['length_matches'] and sq.get('M5'):
                digest = hashlib.md5()
                for start in range(0, sq['LN'], 1024 * 1024):
                    digest.update(fasta.fetch(name, start, min(start + 1024 * 1024, sq['LN'])).upper().encode('ascii'))
                row['reference_md5'] = digest.hexdigest()
                row['md5_matches'] = digest.hexdigest() == sq['M5'].lower()
            else:
                row['md5_matches'] = None
            results.append(row)
    return {'contigs': results, 'all_lengths_match': bool(results) and all(r['length_matches'] for r in results),
            'all_header_md5_present_and_match': bool(results) and all(r['md5_matches'] is True for r in results)}


def inspect_alignment(path, expected_length, reference=None, limit=0):
    import pysam
    from .samples import canonical_contig, GRCH38_LENGTHS
    if expected_length < 1 or limit < 0:
        raise ValueError('Expected length must be positive and limit nonnegative')
    cram = str(path).lower().endswith('.cram')
    if cram and reference is None:
        raise ValueError('CRAM QC requires an explicit decoding reference')
    started = time.perf_counter()
    pysam.quickcheck(str(path))
    if reference and not Path(str(reference) + '.fai').exists():
        pysam.faidx(str(reference))
    counts, lengths, groups, mapq, qualities, bases, clipping, contig_counts = (Counter() for _ in range(8))
    complete = not bool(limit)
    with pysam.AlignmentFile(str(path), 'rc' if cram else 'rb', reference_filename=str(reference) if reference else None) as handle:
        header = handle.header.to_dict()
        ref_check = verify_reference(header, reference) if reference else None
        if ref_check and (not ref_check['all_lengths_match'] or any(r['md5_matches'] is False for r in ref_check['contigs'])):
            raise ValueError('Decoding reference does not match alignment @SQ lengths/MD5s')
        for read in handle.fetch(until_eof=True):
            if limit and counts['primary_reads_with_sequence'] >= limit:
                break
            counts['total_records'] += 1
            counts['secondary_records'] += read.is_secondary
            counts['supplementary_records'] += read.is_supplementary
            counts['all_records_qcfail'] += read.is_qcfail
            # Force sequence and quality access for all records, including excluded alignments.
            sequence, quality = read.query_sequence, read.query_qualities
            if read.is_secondary or read.is_supplementary:
                continue
            counts['primary_records'] += 1
            for key, flag in [('duplicates', read.is_duplicate), ('unmapped', read.is_unmapped),
                              ('qcfail', read.is_qcfail), ('paired', read.is_paired),
                              ('proper_pair', read.is_proper_pair), ('read1', read.is_read1),
                              ('read2', read.is_read2), ('reverse', read.is_reverse),
                              ('mate_unmapped', read.is_paired and read.mate_is_unmapped)]:
                counts['primary_' + key] += flag
            if not read.is_unmapped:
                mapq[read.mapping_quality] += 1
                contig_counts[read.reference_name] += 1
            soft = sum(n for op, n in (read.cigartuples or []) if op == 4)
            hard = sum(n for op, n in (read.cigartuples or []) if op == 5)
            clipping['soft_clipped_reads'] += bool(soft)
            clipping['soft_clipped_bases'] += soft
            clipping['hard_clipped_reads'] += bool(hard)
            clipping['hard_clipped_bases'] += hard
            if sequence is None:
                counts['primary_without_sequence'] += 1
                continue
            counts['primary_reads_with_sequence'] += 1
            lengths[len(sequence)] += 1
            group = read.get_tag('RG') if read.has_tag('RG') else '<missing>'
            if group not in groups:
                groups[group] = Counter()
            groups[group][len(sequence)] += 1
            bases.update(sequence.upper())
            if quality is None:
                counts['primary_sequence_reads_without_quality'] += 1
            else:
                qualities.update(quality)
                counts['quality_length_mismatches'] += len(quality) != len(sequence)
            if counts['total_records'] % 1000000 == 0:
                print(f"QC: {counts['total_records']:,} alignment records decoded", flush=True)
    primary = counts['primary_reads_with_sequence']
    fraction = lengths[expected_length] / primary if primary else 0
    contigs = {canonical_contig(s['SN']): s['LN'] for s in header.get('SQ', [])}
    nqual = sum(qualities.values())
    total_bases = sum(bases.values())
    result = dict(counts)
    for key in ('primary_without_sequence', 'primary_sequence_reads_without_quality', 'quality_length_mismatches'):
        result.setdefault(key, 0)
    result.update({
        'schema_version': 2, 'path': str(path), 'format': 'cram' if cram else 'bam',
        'expected_read_length': expected_length, 'scope': 'full_file' if complete else 'prefix_only',
        'full_decode_completed': complete, 'quickcheck_passed': True,
        'length_histogram': dict(sorted(lengths.items())), 'read_group_lengths': dict(groups),
        'exact_length_fraction': fraction, 'modal_read_length': lengths.most_common(1)[0][0] if lengths else None,
        'read_groups': header.get('RG', []), 'header': header,
        'grch38_primary_lengths_match': all(contigs.get(k) == v for k, v in GRCH38_LENGTHS.items()),
        'reference_verification': ref_check,
        'contig_aliases': {s['SN']: canonical_contig(s['SN']) for s in header.get('SQ', [])},
        'eligible_length_cohort': bool(complete and primary and fraction >= 0.95),
        'read_length_policy': 'At least 95% of primary sequence-bearing records at expected length; no trimming',
        'metric_scope': 'Flag, clipping and mapped-contig/MAPQ counts: all primary records. Length/base/quality metrics: primary records with sequence. Secondary/supplementary counts may overlap.',
        'mapping_quality_histogram_primary_mapped': dict(sorted(mapq.items())),
        'mapq_255_note': '255 means mapping quality unavailable, not high confidence',
        'base_quality_histogram': dict(sorted(qualities.items())), 'quality_bases': nqual,
        'mean_base_quality': sum(q*n for q,n in qualities.items()) / nqual if nqual else None,
        'q20_fraction': sum(n for q,n in qualities.items() if q >= 20) / nqual if nqual else None,
        'q30_fraction': sum(n for q,n in qualities.items() if q >= 30) / nqual if nqual else None,
        'base_counts': dict(bases), 'sequence_bases': total_bases,
        'gc_fraction_all_sequence_bases': (bases['G'] + bases['C']) / total_bases if total_bases else None,
        'n_fraction': bases['N'] / total_bases if total_bases else None,
        'clipping': dict(clipping), 'primary_mapped_records_by_contig': dict(contig_counts),
        'pysam_version': pysam.__version__, 'samtools_version': pysam.__samtools_version__,
        'checked_at': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter() - started,
        'limitations': ['No capture BED coverage/on-target assessment', 'No contamination or biological identity inference',
                        'No duplicate removal, trimming, realignment or filtering', 'Hard-clipped bases cannot be recovered']})
    return result


def compare_archive_statistics(path, qc):
    """Compare the archive's alignment-record total, not unique reads/fragments."""
    import csv
    with open(path) as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    expected = sum(int(row['#_total_reads']) for row in rows)
    return {'path': str(path), 'archive_total_alignment_records': expected,
            'observed_total_alignment_records': qc['total_records'],
            'total_matches': expected == qc['total_records'],
            'archive_rows': rows,
            'note': 'BAS total counts alignment records, including non-primary records; do not compare it with primary read count or interpret it as unique fragments.'}
