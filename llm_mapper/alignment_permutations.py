"""Conditional gene-label permutation null for an existing SW-scored graph."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .alignment import SmithWaterman, SWConfig
from .alignment.smith_waterman import dna, reverse_complement
from .gene_clusters import write_csv
from .neighborhood_alignment import load_inputs
from .preparation.downloads import file_hash, write_json


def make_strata(gc, lengths, degree, bins=5):
    """Marginal quantile bins crossed with exact degree; ties stay together."""
    if bins < 1 or np.any(np.asarray(lengths) <= 0):
        raise ValueError('Positive bins and lengths required')
    boundaries = [np.unique(np.quantile(v, np.arange(1, bins)/bins))
                  for v in (gc, np.log(lengths))]
    keys = np.column_stack([np.searchsorted(b, v, side='right') for b, v in
                           zip(boundaries, (gc, np.log(lengths)))] + [degree])
    unique, labels, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    groups = [np.flatnonzero(labels == i) for i in range(len(unique))]
    return groups, labels, counts[labels], [b.tolist() for b in boundaries]


def permutation(groups, size, seed, index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    assignment = np.arange(size)
    for group in groups:
        assignment[group] = rng.permutation(group)
    return assignment


def gene_means(rows, cols, scores, degree, assignment):
    """assignment[node] = gene; return means in original gene order."""
    sums = np.bincount(rows, weights=scores, minlength=len(degree))
    sums += np.bincount(cols, weights=scores, minlength=len(degree))
    values = np.divide(sums, degree, out=np.full(len(degree), np.nan), where=degree > 0)
    result = np.empty_like(values)
    result[assignment] = values
    return result


def null_statistics(observed, null):
    if null.ndim != 2 or null.shape[1] != len(observed) or len(null) < 2:
        raise ValueError('Need at least two permutations with matching gene order')
    # Small tolerance treats floating summation noise as ties, conservatively.
    exceed = np.sum(null >= observed[None, :] - 1e-12, axis=0)
    p = (exceed + 1)/(len(null) + 1)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    q = np.empty_like(p); q[order] = np.minimum(adjusted, 1)
    return dict(null_mean=null.mean(axis=0), null_sd=null.std(axis=0, ddof=1),
                null_p025=np.quantile(null, .025, axis=0), null_p975=np.quantile(null, .975, axis=0),
                excess_similarity=observed-null.mean(axis=0), exceedances=exceed,
                p_enrichment=p, q_bh_exploratory=q)


def summarize(output, observed, null, nodes, xy, sizes):
    stats = null_statistics(observed, null)
    records = []
    for i, node in enumerate(nodes):
        records.append(dict(gene_symbol=node['gene_symbol'], cluster=int(node['cluster']),
                            stratum_size=int(sizes[i]), restricted_stratum=bool(sizes[i]<5),
                            observed_similarity=float(observed[i]), observed_distance=float(1-observed[i]),
                            **{k: float(v[i]) for k,v in stats.items()}))
    write_csv(output/'gene_sw_null.csv', records)
    summaries = []
    clusters = np.array([int(n['cluster']) for n in nodes])
    for label in ['all'] + list(np.unique(clusters)):
        mask = np.ones(len(nodes), dtype=bool) if label == 'all' else clusters == label
        obs = observed[mask].mean(); values = null[:, mask].mean(axis=1)
        summaries.append(dict(cluster=str(label), genes=int(mask.sum()), observed_mean=float(obs),
                              null_mean=float(values.mean()), excess=float(obs-values.mean()),
                              null_p025=float(np.quantile(values,.025)), null_p975=float(np.quantile(values,.975)),
                              p_enrichment=float((1+sum(values>=obs-1e-12))/(len(values)+1))))
    write_csv(output/'cluster_sw_null.csv', summaries)
    write_json(output/'summary.json', dict(permutations=len(null), minimum_p=1/(len(null)+1),
               genes_positive_excess=int(sum(stats['excess_similarity']>0)),
               genes_p_le_005=int(sum(stats['p_enrichment']<=.05)),
               genes_q_bh_le_005=int(sum(stats['q_bh_exploratory']<=.05)), summaries=summaries,
               caveat='Conditional random-label null; bins only approximately match GC/length. Cluster tests are descriptive post-selection. Gene BH q-values are exploratory; graph dependence and 100-permutation resolution limit inference.'))
    report = ["# Stratified SW null results", "",
              f"Completed {len(null)} GC/length/degree-stratified permutations.", "",
              "| Group | Genes | Observed mean | Null mean | Excess | Enrichment p |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        report.append(f"| {row['cluster']} | {row['genes']} | {row['observed_mean']:.6f} | {row['null_mean']:.6f} | {row['excess']:.6f} | {row['p_enrichment']:.4f} |")
    report += ["", "The all-gene statistic averages gene neighborhood means, giving each gene equal weight.",
               "Cluster results follow original gene membership and are descriptive post-selection comparisons.",
               "", f"Minimum attainable p: {1/(len(null)+1):.6f}. Individual-gene p and BH q values are exploratory.",
               "Null intervals describe the random-label distribution, not uncertainty in biological effects.",
               "See gene_sw_null.csv, cluster_sw_null.csv, sw_null_umap.png and sw_null_qc.png."]
    (output/'report.md').write_text('\n'.join(report)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(19,5.6),layout='constrained')
    limit=max(float(np.quantile(np.abs(stats['excess_similarity']),.99)),1e-8)
    for ax, values, title, cmap, low, high in zip(axes,
            [observed,stats['excess_similarity'],-np.log10(stats['p_enrichment'])],
            ['Observed mean SW similarity','Excess over matched null','Enrichment: −log₁₀ empirical p'],
            ['magma','RdBu_r','viridis'],[0,-limit,0],
            [max(float(np.quantile(observed,.99)),1e-8),limit,np.log10(len(null)+1)]):
        artist=ax.scatter(*xy.T,c=values,s=4,cmap=cmap,vmin=low,vmax=high,linewidths=0,rasterized=True)
        fig.colorbar(artist,ax=ax,shrink=.8,extend='both' if cmap=='RdBu_r' else 'max')
        ax.set(title=title,xlabel='UMAP 1',ylabel='UMAP 2',xticks=[],yticks=[])
    fig.suptitle(f'Sequence similarity beyond GC / CDS length / degree · {len(null)} label permutations\nObserved and excess colors clipped at 99th percentile; p-values unadjusted',fontsize=13)
    for extension in ('png','pdf'):fig.savefig(output/f'sw_null_umap.{extension}',dpi=230)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].hist(null.mean(axis=1),bins=20,color='#64748b');axes[0].axvline(observed.mean(),color='#dc2626',label='Observed')
    axes[0].set(xlabel='Global mean neighborhood SW',ylabel='Permutations',title='Global conditional null');axes[0].legend()
    axes[1].scatter(stats['null_mean'],observed,s=3,alpha=.4,rasterized=True)
    upper=max(observed.max(),stats['null_mean'].max());axes[1].plot([0,upper],[0,upper],color='gray',linestyle='--')
    axes[1].set(xlabel='Null mean',ylabel='Observed mean',title='Gene neighborhood similarity')
    for extension in ('png','pdf'):fig.savefig(output/f'sw_null_qc.{extension}',dpi=220)
    plt.close(fig)


def run(database, graph_directory, fasta, observed_directory, output, permutations=100,
        pilot=2, bins=5, seed=20260929, workers=8, stop_after=None):
    if permutations<2 or not 1<=pilot<=permutations or workers<1 or seed<0:
        raise ValueError('Invalid permutation, pilot, worker or seed settings')
    db, graph, sequences, nodes, xy, signature = load_inputs(database,graph_directory,fasta)
    observed_directory=Path(observed_directory); output=Path(output);output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        observed_manifest=json.loads((observed_directory/'manifest.json').read_text())
        if observed_manifest['status']!='complete' or any(observed_manifest['signature'].get(k)!=v for k,v in signature.items()):
            raise ValueError('Observed SW provenance differs from input')
        config=SWConfig(**observed_manifest['signature']['scoring']);engine=SmithWaterman(config)
        if file_hash(engine.parasail.get_library())!=observed_manifest['signature']['parasail_library_sha256']:
            raise ValueError('Native SW library differs from observed run')
        signature.update(scoring=asdict(config),library_sha256=file_hash(engine.parasail.get_library()),
                         observed_csv_sha256=file_hash(observed_directory/'gene_neighbor_sw.csv'),
                         bins=bins,seed=seed,permutations=permutations,pilot=pilot,method_version=1,
                         null='Gene-following permutations within marginal GC/log-length quantile bins crossed with exact degree')
        with (observed_directory/'gene_neighbor_sw.csv').open() as handle: records=list(csv.DictReader(handle))
        if [r['gene_symbol'] for r in records]!=[n['gene_symbol'] for n in nodes]:raise ValueError('Observed order mismatch')
        observed=np.array([float(r['mean_sw_similarity']) for r in records])
        degree=np.diff(graph.indptr);lengths=np.array([len(s) for s in sequences],dtype=np.int64)
        if (degree==0).any():raise ValueError('Isolated nodes not supported in this null')
        gc=np.array([g['gc_fraction'] for g in db.genes])
        groups,labels,sizes,boundaries=make_strata(gc,lengths,degree,bins)
        manifest_path=output/'manifest.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            if manifest['signature']!=signature:raise ValueError('Resume signature mismatch')
        else:manifest=dict(signature=signature,status='pilot',completed=[])
        write_json(manifest_path,manifest)
        write_csv(output/'strata.csv',[dict(gene_symbol=n['gene_symbol'],stratum=int(labels[i]),
                  stratum_size=int(sizes[i]),degree=int(degree[i]),gc_fraction=float(gc[i]),cds_length=int(lengths[i])) for i,n in enumerate(nodes)])
        write_json(output/'strata_qc.json',dict(bins_per_covariate=bins,boundaries_gc_log_length=boundaries,
                   strata=len(groups),singleton_genes=int(sum(sizes==1)),genes_with_at_least_five_positions=int(sum(sizes>=5)),
                   median_stratum_size=float(np.median(sizes)),degree_matching='exact'))
        upper=sparse.triu(graph,k=1).tocoo();rows,cols=upper.row,upper.col
        # Validate once and reuse reverse complements; exact same native SW kernel/scoring.
        sequences=[dna(s) for s in sequences]; reverse=[reverse_complement(s) for s in sequences]
        self_scores=np.array([config.match*(len(s)-s.count('N')) for s in sequences])
        null=[]
        for record in manifest['completed']:
            path=output/record['file']
            if file_hash(path)!=record['sha256']:raise ValueError('Checkpoint checksum mismatch')
            values=np.load(path,allow_pickle=False)
            if values.shape!=observed.shape or not np.isfinite(values).all():raise ValueError('Invalid checkpoint values')
            null.append(values)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index in range(len(null),permutations):
                start=time.perf_counter();assignment=permutation(groups,len(nodes),seed,index)
                if not np.array_equal(degree[assignment],degree):raise AssertionError('Degree changed')
                source,target=assignment[rows],assignment[cols]
                cells=lengths[source]*lengths[target]*(2 if config.both_strands else 1)
                order=np.argsort(-cells,kind='stable');scores=np.empty(len(rows))
                def align(k):
                    i,j=source[k],target[k]
                    score=engine.raw_score(sequences[i],sequences[j])
                    if config.both_strands:score=max(score,engine.raw_score(sequences[i],reverse[j]))
                    return 2*score/(self_scores[i]+self_scores[j])
                last_update=start
                for offset in range(0,len(order),1024):
                    selected=order[offset:offset+1024];scores[selected]=list(pool.map(align,selected))
                    if time.perf_counter()-last_update>30:
                        print(f'Permutation {index+1}/{permutations}: {offset+len(selected):,}/{len(rows):,} edges',flush=True);last_update=time.perf_counter()
                values=gene_means(rows,cols,scores,degree,assignment)
                if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError('Invalid null scores')
                name=f'permutation_{index:03d}.npy';temporary=output/(name+'.tmp')
                with temporary.open('wb') as handle:np.save(handle,values,allow_pickle=False)
                temporary.replace(output/name);null.append(values)
                record=dict(index=index,file=name,sha256=file_hash(output/name),seconds=time.perf_counter()-start,
                            dp_cells=int(cells.sum()),fixed_gene_fraction=float(np.mean(assignment==np.arange(len(nodes)))),
                            mean_absolute_gc_shift=float(np.mean(abs(gc[assignment]-gc))),
                            mean_absolute_log_length_shift=float(np.mean(abs(np.log(lengths[assignment]/lengths)))))
                manifest['completed'].append(record);manifest['status']='running' if len(null)>=pilot else 'pilot'
                write_json(manifest_path,manifest)
                print(f'Completed permutation {index+1}/{permutations} in {record["seconds"]:.1f}s',flush=True)
                if len(null)==pilot:
                    timings=[r['seconds'] for r in manifest['completed']]
                    estimate=dict(pilot_permutations=pilot,mean_seconds=float(np.mean(timings)),
                                  projected_total_hours=float(np.mean(timings)*permutations/3600),
                                  projected_remaining_hours=float(np.mean(timings)*(permutations-pilot)/3600),
                                  note='Pilot permutations are included in the prespecified total; runtime includes alignment and aggregation. No outcome-based stopping.')
                    write_json(output/'pilot.json',estimate);print('PILOT '+json.dumps(estimate),flush=True)
                if stop_after is not None and len(null)>=stop_after:return manifest
        summarize(output,observed,np.array(null),nodes,xy,sizes)
        manifest.update(status='complete',total_permutation_seconds=sum(r['seconds'] for r in manifest['completed']))
        write_json(manifest_path,manifest)
        return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('database','graph_directory','fasta','observed_directory','output'):parser.add_argument(name)
    parser.add_argument('--permutations',type=int,default=100);parser.add_argument('--pilot',type=int,default=2)
    parser.add_argument('--bins',type=int,default=5);parser.add_argument('--seed',type=int,default=20260929)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--stop-after',type=int)
    run(**vars(parser.parse_args()))


if __name__=='__main__':main()
