"""CPU-only diagnostics from persisted contrastive training logs; no weights loaded."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    args=parser.parse_args();root=args.experiment;t=root/'training';out=t/'diagnostics';out.mkdir(exist_ok=True)
    manifest=json.loads((t/'training_manifest.json').read_text());settings=manifest['signature']['settings']
    train=pd.DataFrame(read_jsonl(t/'training_log.jsonl'))
    if train.optimizer_step.duplicated().any() or not np.isfinite(train[['loss','gradient_norm']]).all().all():
        raise ValueError('Training log has duplicate steps or nonfinite values')
    if not np.all(np.diff(train.optimizer_step)==1):raise ValueError('Nonconsecutive training updates')
    train['loss_mean50']=train.loss.rolling(50,min_periods=50).mean()
    val={}
    sources=list((t/'metrics_history').glob('*.json'))+list(t.glob('checkpoint-*/checkpoint_manifest.json'))
    for p in sources:
        state=json.loads(p.read_text()).get('state',{});v=state.get('validation',{})
        if 'mean_info_nce' in v:val[state['optimizer_step']]={'optimizer_step':state['optimizer_step'],**v}
    for r in read_jsonl(t/'validation_log.jsonl'):
        if 'mean_info_nce' in r['validation']:val[r['optimizer_step']]={'optimizer_step':r['optimizer_step'],**r['validation']}
    validation=pd.DataFrame([val[k] for k in sorted(val)])
    mem=[];segment=0;previous=None
    for r in read_jsonl(t/'memory_log.jsonl'):
        if previous is not None and r['elapsed_seconds']<previous:segment+=1
        previous=r['elapsed_seconds']
        row={k:r[k] for k in ['microstep','optimizer_step','elapsed_seconds']};row['invocation_segment']=segment
        row['total_anchor_bp']=sum(r['anchor_lengths']);row['max_anchor_bp']=max(r['anchor_lengths'])
        for stage in ['before','after_gc','after_cache_clear']:
            for key in ['tensor_bytes','driver_bytes','rss_bytes']:row[f'{stage}_{key}']=r[stage][key]/1e9
        mem.append(row)
    memory=pd.DataFrame(mem)
    memory['microbatch_seconds']=memory.groupby('invocation_segment').elapsed_seconds.diff()
    memory['driver_reclaimed_gb']=memory.before_driver_bytes-memory.after_cache_clear_driver_bytes
    # Deltas include intervening optimizer/checkpoint/validation work, not just kernels.
    for name,df in [('training_metrics',train),('validation_metrics',validation),('memory_metrics',memory)]:df.to_csv(out/f'{name}.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':180})
    figures=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/f'{name}.png');fig.savefig(out/f'{name}.pdf');plt.close(fig);figures.append(name)
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    axes[0].plot(train.optimizer_step,train.loss,color='#8aabc5',alpha=.45,lw=.65,label='Per-update training loss')
    axes[0].plot(train.optimizer_step,train.loss_mean50,color='#176b91',lw=2,label='Trailing 50-update mean')
    axes[0].plot(validation.optimizer_step,validation.mean_info_nce,'o-',color='#d97432',label='Fixed-subset validation loss')
    axes[0].set(ylabel='InfoNCE cross entropy',title='GB.DNA-300M · exon-contained contrastive fine-tuning');axes[0].legend()
    axes[1].plot(validation.optimizer_step,validation.within_batch_accuracy*100,'o-',color='#34856b')
    axes[1].set(xlabel='Optimizer update',ylabel='Validation accuracy (%)',ylim=(0,100),title='25 fixed triplets · within-batch retrieval, not WES mapping')
    save(fig,'01_learning_curves')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(train.optimizer_step,train.gradient_norm,lw=.7,color='#7853a2');axes[0].axhline(1,color='black',ls='--',label='Clip threshold = 1')
    axes[0].set(xlabel='Optimizer update',ylabel='Pre-clipping gradient norm');axes[0].legend()
    axes[1].hist(train.gradient_norm,bins=40,color='#7853a2',alpha=.8);axes[1].set(xlabel='Pre-clipping gradient norm',ylabel='Updates')
    save(fig,'02_gradient_diagnostics')
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    x=memory.microstep/settings['gradient_accumulation']
    for key,label,color in [('before_driver_bytes','Driver before cleanup','#bb7d35'),('after_cache_clear_driver_bytes','Driver after cleanup','#176b91'),('after_cache_clear_tensor_bytes','Live tensors after cleanup','#34856b')]:axes[0].plot(x,memory[key],lw=.6,label=label,color=color)
    axes[0].set(ylabel='GB (decimal)',title='Post-backward memory samples · within-operation peaks not measured');axes[0].legend()
    axes[1].plot(x,memory.driver_reclaimed_gb,lw=.6,color='#7853a2');axes[1].set(xlabel='Approximate optimizer-update position',ylabel='Driver GB reclaimed',title='Garbage collection + cache release; RSS is not added to unified GPU memory')
    save(fig,'03_memory_cleanup')
    runtime=memory.dropna(subset=['microbatch_seconds'])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(runtime.microstep/settings['gradient_accumulation'],runtime.microbatch_seconds,lw=.6,color='#176b91');axes[0].set(xlabel='Approximate optimizer-update position',ylabel='Seconds between memory samples',title='Includes intervening checkpoint/validation overhead')
    axes[1].scatter(runtime.total_anchor_bp,runtime.microbatch_seconds,s=5,alpha=.25,color='#34856b');axes[1].set(xscale='log',yscale='log',xlabel='Total anchor CDS bases in microbatch',ylabel='Seconds between samples',title='Input length and observed runtime')
    save(fig,'04_runtime_and_length')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].scatter(memory.max_anchor_bp,memory.before_driver_bytes,s=5,alpha=.25,color='#bb7d35');axes[0].set(xscale='log',xlabel='Longest anchor CDS (bp)',ylabel='Driver GB before cleanup')
    axes[1].plot(train.optimizer_step,np.full(len(train),settings['learning_rate']),color='#176b91');axes[1].set(xlabel='Optimizer update',ylabel='Configured learning rate',title='Constant AdamW learning rate (no scheduler)');axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    save(fig,'05_length_memory_and_learning_rate')
    fig,ax=plt.subplots(figsize=(11,4))
    for folder,color in [('attempt_01_failed','#bb7d35'),('attempt_02_failed','#7853a2')]:
        p=t/folder/'training_log.jsonl'
        if p.exists():
            a=pd.DataFrame(read_jsonl(p));ax.plot(a.optimizer_step,a.loss.rolling(25,min_periods=25).mean(),label=folder,lw=1.2,color=color,alpha=.75)
    ax.plot(train.optimizer_step,train.loss.rolling(25,min_periods=25).mean(),label='Completed trajectory',lw=1,color='#176b91')
    ax.set(xlabel='Optimizer update',ylabel='Trailing 25-update mean loss',title='Attempts overlaid at matching updates · repeated work is not extra training');ax.legend()
    save(fig,'06_attempt_comparison')
    summary=dict(status=manifest['status'],optimizer_updates=int(train.optimizer_step.iloc[-1]),microsteps=int(train.microstep.iloc[-1]),first50_mean_loss=float(train.loss.head(50).mean()),last50_mean_loss=float(train.loss.tail(50).mean()),gradient_clipping_fraction=float((train.gradient_norm>1).mean()),final_validation=val[max(val)],lowest_sampled_validation_loss=val[min(val,key=lambda k:val[k]['mean_info_nce'])],memory_samples=len(memory),sampled_driver_before_max_gb=float(memory.before_driver_bytes.max()),sampled_live_tensor_max_gb=float(memory.before_tensor_bytes.max()),median_microbatch_interval_seconds=float(runtime.microbatch_seconds.median()),memory_log_invocations=int(memory.invocation_segment.max()+1),figures=figures)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    inputs=[t/'training_log.jsonl',t/'validation_log.jsonl',t/'memory_log.jsonl',t/'training_manifest.json',*sources,*t.glob('attempt_*_failed/training_log.jsonl')]
    (out/'provenance.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2)+'\n')
    report=['# Contrastive training diagnostics','',f"Status: **{summary['status']}**. {summary['optimizer_updates']:,} optimizer updates; {summary['microsteps']:,} microbatches.",'',f"Mean training loss: first 50 updates **{summary['first50_mean_loss']:.3f}**, last 50 **{summary['last50_mean_loss']:.3f}**.",f"Final validation: InfoNCE **{val[max(val)]['mean_info_nce']:.3f}**, within-batch accuracy **{val[max(val)]['within_batch_accuracy']:.0%}**.",f"Pre-clipping gradient norm exceeded 1 on **{summary['gradient_clipping_fraction']:.1%}** of updates.",'','## Interpretation and limits','','Training loss decreased, but validation fluctuated and did not improve monotonically. The final weights are the final epoch checkpoint, not a selected best-validation model. Low training loss alone does not establish improved held-out WES retrieval.','', 'Validation uses 25 fixed triplets with within-batch candidate sets. It is not evaluation of all 2,500 validation triplets, all HGNC genes, or WES reads. The implementation averages batch losses equally, including the final one-triplet batch; training and validation candidate counts therefore differ slightly. Accuracy is triplet-weighted.','', 'Training loss is recorded before each optimizer update and averaged across its accumulation window. The last update has fewer examples. Curves use the completed trajectory once; discarded attempts are overlaid separately and must not be concatenated as additional epochs.','', 'Memory profiling began at update 250 in recovery attempt 03. Samples are after backward, not peak memory during forward/backward. Driver allocation includes backend/cached allocations; RSS overlaps unified memory. Runtime intervals reset at automatic resume and include intervening optimizer, checkpoint and validation work. Learning rate is reconstructed from the constant configuration, not independently sampled.','', 'No per-triplet training accuracy, pre-training validation baseline, weight norms or parameter-update norms were logged; these cannot be reconstructed honestly from these logs.','', '## Plots','']
    for name in figures:report += [f'### {name.replace("_"," ")}',f'![{name}]({name}.png)',f'[PDF]({name}.pdf)','']
    report += ['## Machine-readable outputs','','`training_metrics.csv`, `validation_metrics.csv`, `memory_metrics.csv`, `summary.json`, and `provenance.json`. Regenerate from the project root:','','```bash',f'.venv/bin/python -m scripts.report_training_diagnostics {root.as_posix()}','```','']
    (out/'README.md').write_text('\n'.join(report));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
