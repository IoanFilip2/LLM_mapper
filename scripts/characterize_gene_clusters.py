"""Test transcript/length and HGNC-family enrichment in selected gene clusters."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import numpy as np
from scipy.stats import fisher_exact, false_discovery_control, mannwhitneyu, hypergeom
from scipy.special import logsumexp
from llm_mapper.preparation.downloads import file_hash, write_json
from llm_mapper.gene_clusters import write_csv


def enrichment(cluster, feature):
    a=int(np.sum(cluster & feature));b=int(np.sum(cluster & ~feature))
    c=int(np.sum(~cluster & feature));d=int(np.sum(~cluster & ~feature))
    odds,p=fisher_exact([[a,b],[c,d]],alternative='greater')
    log10p=float(logsumexp(hypergeom.logpmf(np.arange(a,min(a+b,a+c)+1),a+b+c+d,a+c,a+b))/np.log(10))
    return {'cluster_positive':a,'cluster_total':a+b,'rest_positive':c,'rest_total':c+d,
            'odds_ratio':float(odds) if np.isfinite(odds) else None,
            'odds_ratio_nonfinite':str(odds) if not np.isfinite(odds) else '', 'p':float(p),'log10_p':min(0.,log10p)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database');parser.add_argument('assignments');parser.add_argument('hgnc');parser.add_argument('output')
    parser.add_argument('--clusters',type=int,nargs='+',default=[3,4]);parser.add_argument('--top-n',type=int,default=50)
    args=parser.parse_args();out=Path(args.output)
    if args.top_n < 1 or len(args.clusters) != len(set(args.clusters)):
        raise ValueError('top-n must be positive and cluster IDs unique')
    if json.loads((Path(args.database)/'manifest.json').read_text())['status'] != 'complete':
        raise ValueError('A completed database is required')
    if out.exists():raise FileExistsError('Choose a new output directory')
    with sqlite3.connect(Path(args.database)/'metadata.sqlite') as con:
        metadata={g['gene_symbol']:g for (raw,) in con.execute('SELECT metadata FROM genes') for g in [json.loads(raw)]}
    with open(args.assignments) as handle:rows=list(csv.DictReader(handle))
    if len(rows)!=len(metadata) or {r['gene_symbol'] for r in rows}!=set(metadata):raise ValueError('Gene sets differ')
    with open(args.hgnc) as handle:hgnc={r['hgnc_id']:r for r in csv.DictReader(handle,delimiter='\t')}
    annotation=[hgnc[metadata[r['gene_symbol']]['hgnc_id']] for r in rows]
    groups=[set(a['gene_group'].split('|'))-{''} for a in annotation]
    methods=np.array([metadata[r['gene_symbol']]['selection_method'] for r in rows])
    length=np.array([int(r['cds_length']) for r in rows])
    spans=np.array([int(metadata[r['gene_symbol']]['end'])-int(metadata[r['gene_symbol']]['start'])+1 for r in rows])
    labels=np.array([int(r['cluster']) for r in rows])
    features={'selected_non_MANE':methods!='MANE_Select',
              'no_MANE_in_HGNC_snapshot':np.array([not bool(a['mane_select']) for a in annotation]),
              'CDS_over_3998_bp':length>3998,'CDS_at_least_10000_bp':length>=10000,
              'CDS_at_least_20000_bp':length>=20000,'genomic_span_at_least_1Mb':spans>=1000000}
    out.mkdir(parents=True)
    stats=[];families=[];descriptions=[];tops=[]
    bg=Counter(g for gs in groups for g in gs)
    for cluster in args.clusters:
        selected=labels==cluster
        if not selected.any() or selected.all():raise ValueError('Cluster or complement empty')
        top=sorted(np.flatnonzero(selected),key=lambda i:int(rows[i]['centrality_rank']))[:args.top_n]
        for name,feature in features.items():stats.append({'cluster':cluster,'feature':name,**enrichment(selected,feature)})
        for group,count in sorted(bg.items()):
            if count>=5:
                present=np.array([group in gs for gs in groups])
                families.append({'cluster':cluster,'gene_group':group,'background_count':count,
                    **enrichment(selected,present), 'cluster_genes':';'.join(rows[i]['gene_symbol'] for i in np.flatnonzero(selected & present))})
        u,p=mannwhitneyu(length[selected],length[~selected],alternative='two-sided')
        descriptions.append({'cluster':cluster,'genes':int(selected.sum()),'selection_counts':dict(Counter(methods[selected])),
            'CDS_median':float(np.median(length[selected])),'rest_CDS_median':float(np.median(length[~selected])),
            'CDS_p95':float(np.quantile(length[selected],.95)),'CDS_max':int(length[selected].max()),
            'CDS_length_Mann_Whitney_two_sided_p':float(p),'CDS_length_rank_biserial':float(2*u/(selected.sum()*(~selected).sum())-1),
            'top_n':len(top),'top_selection_counts':dict(Counter(methods[top])),
            'top_CDS_median':float(np.median(length[top])),'top_CDS_min':int(length[top].min()),'top_CDS_max':int(length[top].max()),
            'top_feature_counts':{k:int(v[top].sum()) for k,v in features.items()},
            'top_gene_groups':dict(Counter(g for i in top for g in groups[i]).most_common()),
            'top_genes': [rows[i]['gene_symbol'] for i in top]})
        for i in top:
            m=metadata[rows[i]['gene_symbol']];a=annotation[i]
            tops.append({**rows[i],'gene_name':a['name'],'selection_method':m['selection_method'],
                'gene_groups':a['gene_group'],'hgnc_mane_select':a['mane_select'],
                'chromosome':m['chrom'],'genomic_span':int(spans[i]),'context_count':m['window_count']})
    for table in [stats,families]:
        for row,q in zip(table,false_discovery_control([r['p'] for r in table],method='bh')):row['q_BH']=float(q)
        order=np.argsort([r['log10_p'] for r in table],kind='stable')
        ordered=np.array([table[i]['log10_p'] for i in order])+np.log10(len(table)/np.arange(1,len(table)+1))
        adjusted=np.minimum(0.,np.minimum.accumulate(ordered[::-1])[::-1])
        for i,q in zip(order,adjusted):table[i]['log10_q_BH']=float(q)
    for row,q in zip(descriptions,false_discovery_control([r['CDS_length_Mann_Whitney_two_sided_p'] for r in descriptions])):
        row['CDS_length_Mann_Whitney_q_BH']=float(q)
    families.sort(key=lambda r:(r['cluster'],r['log10_q_BH'],r['log10_p']))
    write_csv(out/'annotation_length_enrichment.csv',stats);write_csv(out/'hgnc_family_enrichment.csv',families)
    write_csv(out/'top50_annotated.csv',tops)
    report={'background_genes':len(rows),'background_selection_counts':dict(Counter(methods)),
            'background_feature_counts':{k:int(v.sum()) for k,v in features.items()},'clusters':descriptions,
            'method':'One-sided Fisher enrichment: each full cluster versus all other embedded genes; BH over all cluster-feature tests. HGNC families tested separately with BH over both clusters and all families having >=5 embedded members. Top-50 counts descriptive only. CDS length Mann-Whitney two-sided, BH across requested clusters.',
            'feature_tests':len(stats),'family_tests':len(families),
            'inputs_sha256':{Path(p).name:file_hash(p) for p in [args.assignments,args.hgnc]},
            'database_signature':json.loads((Path(args.database)/'manifest.json').read_text())['signature']}
    write_json(out/'analysis.json',report)
    print(json.dumps(report,indent=2))
    print('FEATURE TESTS',json.dumps(stats,indent=2))
    for c in args.clusters:
        print('TOP FAMILIES',c,json.dumps([{k:v for k,v in r.items() if k!='cluster_genes'} for r in families if r['cluster']==c][:12],indent=2))


if __name__=='__main__':main()
