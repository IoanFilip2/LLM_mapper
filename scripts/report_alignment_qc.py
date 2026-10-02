"""Publish an existing alignment QC JSON and a readable summary without rerunning QC."""
import argparse
import json
from pathlib import Path
import shutil


def render(report, json_name, sample):
    lines = [f'# {sample}: direct CRAM QC report', '',
             f"Completed: {report['checked_at']}. Source format: **{report['format']}**.", '',
             f"Manifest readiness gate: **{'PASS' if report.get('ready_for_downstream') else 'NOT PASSED'}**.", '',
             'No intermediate BAM was created for this workflow. All alignment processing used pysam.', '',
             f'[Complete machine-readable report, including header and every reference MD5 comparison]({json_name}).', '',
             '[Reproduction workflow and metric definitions](na19239_verification.md).', '',
             '## Scalar measurements and provenance', '', '| Metric | Value |', '|---|---|']
    primary = report['primary_records']
    mapped = primary - report['primary_unmapped']
    mapq0 = report['mapping_quality_histogram_primary_mapped'].get('0', 0)
    summary = ['', '## Key findings', '', '| Measurement | Result |', '|---|---:|',
               f"| All alignment records | {report['total_records']:,} |",
               f"| Primary records with sequence | {report['primary_reads_with_sequence']:,} |",
               f"| Exactly {report['expected_read_length']} bp | {report['exact_length_fraction']:.4%} of primary sequence-bearing records |",
               f"| Mapped primary records | {mapped:,} ({mapped / primary:.4%}) |",
               f"| Duplicate-flagged primary records | {report['primary_duplicates']:,} ({report['primary_duplicates'] / primary:.4%}) |",
               f"| MAPQ 0 among mapped primary records | {mapq0:,} ({mapq0 / mapped:.4%}) |",
               f"| Mean available base quality | {report['mean_base_quality']:.4f} Phred |",
               f"| Q20 / Q30 bases | {report['q20_fraction']:.4%} / {report['q30_fraction']:.4%} |",
               f"| Primary stored sequence bases | {report['sequence_bases']:,} |",
               f"| QC elapsed time | {report['seconds'] / 60:.2f} minutes |", '',
               'Duplicate and low-MAPQ counts are relevant to later sampling and truth assignment; ',
               'they have not been silently excluded. Balanced read1/read2 counts do not prove ',
               'unique fragment counts or complete mate matching. Secondary/supplementary flags ',
               'can overlap, so their counts must not simply be added to primary records.', '']
    index = lines.index('## Scalar measurements and provenance')
    lines[index:index] = summary
    for key, value in report.items():
        if not isinstance(value, (dict, list)):
            lines.append(f'| {key} | {str(value).replace(chr(10), " ").replace("|", "/")} |')
    lines += ['', '## Distributions and structured measurements', '']
    for key, value in report.items():
        if not isinstance(value, (dict, list)) or key in {'header', 'reference_verification', 'contig_aliases'}:
            continue
        lines += [f'### {key}', '', '```json', json.dumps(value, indent=2), '```', '']
    ref = report.get('reference_verification')
    if ref:
        lines += ['## Reference sequence verification', '',
                  f"- Contigs checked: {len(ref['contigs']):,}",
                  f"- All contig names/lengths match: {ref['all_lengths_match']}",
                  f"- All header MD5s present and matching: {ref['all_header_md5_present_and_match']}", '',
                  'Per-contig results, the complete SAM header and contig aliases are retained in the linked JSON.', '']
    lines += ['## Interpretation', '',
              'Readiness is an acquisition gate, not evidence of biological mapping accuracy. ',
              'The length fraction includes all primary sequence-bearing records, including duplicates, ',
              'QC-failed and unmapped records. Quality, mapping and flag distributions are descriptive; ',
              'no study-specific filtering thresholds have been applied. NA20314 and HG00419 remain deferred.', '']
    return '\n'.join(line.rstrip() for line in lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--sample', required=True)
    parser.add_argument('--output', required=True, help='Output basename, without extension')
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text())
    base = Path(args.output)
    base.parent.mkdir(parents=True, exist_ok=True)
    output_json = base.with_suffix('.json')
    shutil.copyfile(args.input, output_json)
    base.with_suffix('.md').write_text(render(report, output_json.name, args.sample))


if __name__ == '__main__':
    main()
