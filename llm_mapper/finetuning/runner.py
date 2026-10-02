"""Explicit pilot versus training entry points; neither command modifies the base model."""
import argparse
import gc
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import psutil
import torch
from .data import TripletData,prepare
from .model import ContrastiveEncoder,info_nce,negative_mask
from ..embedders import EmbeddingConfig
from ..benchmark import MemorySampler
from ..devices import synchronize
from ..preparation.downloads import file_hash,write_json

DEFAULTS=dict(seed=42,learning_rate=1e-5,weight_decay=.01,temperature=.05,microbatch=2,
              gradient_accumulation=8,epochs=1,checkpointing=True,checkpoint_every=250,keep_checkpoints=2,mps_memory_fraction=.6,max_run_hours=None,validation_every=None,memory_profile=False,mps_cleanup=False)


# Operational changes may accompany a resume; scientific settings remain strict.
OPERATIONAL_SETTINGS={'checkpoint_every','keep_checkpoints','validation_every','memory_profile','mps_cleanup','max_run_hours'}


def compatible_signature(a,b):
    def scientific(signature):
        result=dict(signature)
        result['settings']={k:v for k,v in signature['settings'].items() if k not in OPERATIONAL_SETTINGS}
        return result
    return scientific(a)==scientific(b)


def memory_record(device,cleanup=False):
    def snapshot():
        record={'rss_bytes':psutil.Process().memory_info().rss}
        if device=='mps':
            torch.mps.synchronize()
            record.update(tensor_bytes=torch.mps.current_allocated_memory(),driver_bytes=torch.mps.driver_allocated_memory())
        return record
    result={'before':snapshot()}
    if cleanup:
        result['gc_collected']=gc.collect()
        result['after_gc']=snapshot()
        if device=='mps':torch.mps.empty_cache()
        result['after_cache_clear']=snapshot()
    return result


def seed_everything(seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='mps':torch.mps.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)


def setup(data,device,settings):
    seed_everything(settings['seed'],device)
    if device=='mps':torch.mps.set_per_process_memory_fraction(settings['mps_memory_fraction'])
    model=ContrastiveEncoder(EmbeddingConfig(**data.manifest['base_model_config']),device,
                             data.manifest['chunk_length'],settings['checkpointing'])
    if model.adapter.fingerprint!=data.manifest['base_model_fingerprint']:raise ValueError('Base model fingerprint changed')
    model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'],weight_decay=settings['weight_decay'],foreach=False)
    return model,optimizer


def one_backward(model,batch,temperature,scale=1.):
    anchors,fragments=model(batch);loss=info_nce(anchors,fragments,negative_mask(batch),temperature)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    (loss*scale).backward()
    return float(loss.detach().cpu())


def checked_step(model,optimizer):
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    return float(norm.detach().cpu())


def pilot(dataset,output,device='mps',microbatch=2,repeats=2):
    """Small length-stratified training benchmark; pilot updates are discarded."""
    if not 1<=microbatch<=8 or not 1<=repeats<=3:raise ValueError('Pilot is bounded to batch 1–8 and 1–3 repeats')
    data=TripletData(dataset);output=Path(output)
    if device=='mps':torch.mps.set_per_process_memory_fraction(.6)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a new pilot directory')
    output.mkdir(parents=True,exist_ok=True);settings={**DEFAULTS,'microbatch':microbatch}
    write_json(output/'status.json',dict(status='loading',full_training_started=False))
    model,optimizer=setup(data,device,settings)
    train_ids=np.unique(data.pairs[:,0]);lengths=np.array([len(data.genes[i]['sequence']) for i in train_ids]);order=np.argsort(lengths,kind='stable')
    # Equal-count length strata cover the selected training population, plus an
    # explicit worst-length stress step. Uniform gene counts match 200 pairs/gene.
    strata=np.array_split(order,5);rng=np.random.default_rng(settings['seed']+1);records=[]
    warm_indices=np.arange(microbatch);batch=data.batch(warm_indices)
    with MemorySampler(device) as memory:
        start=time.perf_counter();loss=one_backward(model,batch,settings['temperature']);grad=checked_step(model,optimizer);synchronize(device)
    write_json(output/'warmup.json',dict(seconds=time.perf_counter()-start,loss=loss,gradient_norm=grad,
                                       peak_rss_bytes=memory.peak_rss,peak_device_bytes=memory.peak_device))
    for label,indices in [(str(i),part) for i,part in enumerate(strata)]+[('longest',order[-1:])]:
        for repeat in range(repeats if label!='longest' else 1):
            chosen=train_ids[rng.choice(indices,size=microbatch,replace=len(indices)<microbatch)]
            pair_indices=[int(rng.choice(np.flatnonzero(data.pairs[:,0]==g))) for g in chosen]
            batch=data.batch(pair_indices)
            write_json(output/'status.json',dict(status='measuring',stratum=label,repeat=repeat,full_training_started=False))
            try:
                with MemorySampler(device) as memory:
                    synchronize(device);start=time.perf_counter();loss=one_backward(model,batch,settings['temperature']);synchronize(device);backward_seconds=time.perf_counter()-start
                    start=time.perf_counter();grad=checked_step(model,optimizer);synchronize(device);step_seconds=time.perf_counter()-start
                record=dict(stratum=label,repeat=repeat,pair_indices=pair_indices,gene_lengths=list(map(len,batch['anchors'])),
                            fragment_lengths=list(map(len,batch['positives']+batch['negatives'])),
                            loss=loss,gradient_norm=grad,forward_backward_seconds=backward_seconds,optimizer_seconds=step_seconds,
                            peak_rss_bytes=memory.peak_rss,peak_device_bytes=memory.peak_device)
                records.append(record);write_json(output/'measurements.json',records);print(json.dumps(record),flush=True)
            except RuntimeError as error:
                write_json(output/'status.json',dict(status='failed',stratum=label,error=str(error),full_training_started=False));raise
    sampled=[r for r in records if r['stratum']!='longest'];backward=np.mean([r['forward_backward_seconds'] for r in sampled]);step=np.mean([r['optimizer_seconds'] for r in sampled])
    microsteps=math.ceil(len(data.pairs)/microbatch);optimizer_steps=math.ceil(microsteps/settings['gradient_accumulation'])
    # Pilot optimizes each step to allocate Adam states; production accumulates gradients.
    estimate=backward*microsteps+step*optimizer_steps
    total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    report=dict(status='complete',mode='pilot_only',full_training_started=False,pilot_weights_discarded=True,
                dataset_manifest_sha256=data.signature,device=device,settings=settings,triplets=len(data.pairs),
                trainable_parameters=trainable,total_parameters=total,parameter_gradient_adam_floor_bytes=4*(total+3*trainable),
                physical_memory_bytes=psutil.virtual_memory().total,available_memory_at_end_bytes=psutil.virtual_memory().available,
                peak_rss_bytes=max(r['peak_rss_bytes'] for r in records),peak_device_bytes=max(r['peak_device_bytes'] for r in records),
                memory_note='Sampled process RSS and MPS driver allocations overlap in unified memory; do not sum. Peaks include optimizer states but may miss brief transients. Model load excluded.',
                measured_microsteps=len(records)+1,mean_forward_backward_seconds_per_microbatch=float(backward),mean_optimizer_seconds=float(step),mps_memory_fraction_cap=.6 if device=='mps' else None,
                microsteps_per_epoch=microsteps,optimizer_steps_per_epoch=optimizer_steps,
                projected_training_compute_hours_per_epoch=float(estimate/3600),
                projected_compute_hours_range=[float(estimate*.7/3600),float(estimate*1.5/3600)],
                runtime_caveat='Coarse projection from five equal-count gene-length strata, two samples per stratum. Excludes validation, checkpoint I/O and model loading; thermal/load changes and tail variability remain. Worst-length stress step reported separately.',
                measurements=records)
    write_json(output/'report.json',report);write_json(output/'status.json',dict(status='complete',full_training_started=False))
    return report


def save_checkpoint(model,optimizer,output,state):
    output=Path(output);name=f'checkpoint-{state["optimizer_step"]:06d}';path=output/name
    if path.exists():raise FileExistsError('Checkpoint exists; refusing to overwrite')
    import tempfile
    temporary=Path(tempfile.mkdtemp(prefix=name+'.incomplete-',dir=output))
    model.save_encoder(temporary)
    training=dict(state=state,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                  numpy_rng=np.random.get_state(),python_rng=random.getstate())
    if model.device=='mps':training['device_rng']=torch.mps.get_rng_state()
    if model.device=='cuda':training['device_rng']=torch.cuda.get_rng_state_all()
    torch.save(training,temporary/'training_state.pt')
    inference=asdict(model.adapter.config);inference.update(model=str(path),revision=f'contrastive-step-{state["optimizer_step"]}')
    write_json(temporary/'inference_config.json',inference)
    write_json(temporary/'checkpoint_manifest.json',dict(status='complete',state=state,
               weight_sha256=file_hash(temporary/'model.safetensors'),training_state_sha256=file_hash(temporary/'training_state.pt'),
               base_model_fingerprint=model.adapter.fingerprint,kind='full_encoder_contrastive_finetune',
               requires_new_reference_vectors_and_umap=True))
    temporary.rename(path);write_json(output/'latest.json',dict(checkpoint=name))
    return path


def prune_checkpoints(output,keep,signature):
    """Retain the newest complete checkpoints created by this exact experiment."""
    import shutil
    complete=[]
    for path in Path(output).glob('checkpoint-*'):
        if not path.name.removeprefix('checkpoint-').isdigit() or path.is_symlink():continue
        manifest=path/'checkpoint_manifest.json'
        if not manifest.exists():continue
        m=json.loads(manifest.read_text())
        if compatible_signature(m['state']['signature'],signature):complete.append(path)
    for path in sorted(complete)[:-keep]:shutil.rmtree(path)


def validation(model,data,settings,max_batches=50):
    model.eval();values=[];correct=0;total=0
    rng=np.random.default_rng(settings['seed']+1000)
    genes=np.unique(data.pairs[:,0]);rng.shuffle(genes)
    # Round-robin over genes, not a prefix containing only the first few genes.
    indices=[int(rng.choice(np.flatnonzero(data.pairs[:,0]==g))) for g in genes]
    for start in range(0,min(len(indices),max_batches*settings['microbatch']),settings['microbatch']):
        with torch.no_grad():
            batch=data.batch(indices[start:start+settings['microbatch']])
            anchors,fragments=model(batch);mask=negative_mask(batch).to(model.device)
            values.append(float(info_nce(anchors,fragments,mask,settings['temperature']).cpu()))
            labels=torch.arange(len(anchors),device=model.device);scores=(anchors@fragments.T).masked_fill(~mask,float('-inf'))
            correct+=int((scores.argmax(1)==labels).sum());total+=len(anchors)
    model.train()
    return dict(mean_info_nce=float(np.mean(values)),within_batch_accuracy=correct/total,triplets=total,
                note='Small fixed validation subset; within-batch contrastive accuracy is not whole-reference mapping accuracy')


def train(dataset,output,device='mps',config=None,resume=None):
    """Full training is explicit; this function is never called by pilot."""
    run_started=time.perf_counter()
    data=TripletData(dataset);val=TripletData(dataset,'validation');settings={**DEFAULTS,**(json.loads(Path(config).read_text()) if config else {})}
    if set(settings)-set(DEFAULTS):raise ValueError('Unknown training setting')
    if settings['microbatch']<1 or settings['gradient_accumulation']<1 or settings['epochs']<1 or settings['checkpoint_every']<1 or settings['keep_checkpoints']<1:raise ValueError('Invalid training settings')
    if settings['validation_every'] is not None and settings['validation_every']<1:raise ValueError('Invalid validation interval')
    if not 0<settings['mps_memory_fraction']<=1:raise ValueError('Invalid MPS memory fraction')
    if settings['max_run_hours'] is not None and (not math.isfinite(settings['max_run_hours']) or settings['max_run_hours']<=0):raise ValueError('Runtime budget must be positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest_path=output/'training_manifest.json';signature=dict(dataset_sha256=data.signature,settings=settings,device=device)
        if manifest_path.exists():
            old=json.loads(manifest_path.read_text())
            if not compatible_signature(old['signature'],signature):raise ValueError('Training resume signature mismatch')
            if resume is None:raise ValueError('Existing experiment requires explicit --resume checkpoint path')
        elif any(p.name!='writer.lock' for p in output.iterdir()):raise FileExistsError('Training output must be empty')
        model,optimizer=setup(data,device,settings);start_epoch=0;start_microstep=0;optimizer_step=0;path=Path(resume) if resume else None
        if resume:
            checkpoint=Path(resume);cm=json.loads((checkpoint/'checkpoint_manifest.json').read_text());state=cm['state']
            if not compatible_signature(state['signature'],signature):raise ValueError('Checkpoint belongs to another experiment')
            if file_hash(checkpoint/'training_state.pt')!=cm['training_state_sha256'] or file_hash(checkpoint/'model.safetensors')!=cm['weight_sha256']:raise ValueError('Checkpoint checksum mismatch')
            from safetensors.torch import load_file
            weights=load_file(str(checkpoint/'model.safetensors'))
            current=model.model.state_dict();mapped={k:weights[k.replace('.attention.self.original.','.attention.self.')] for k in current};model.model.load_state_dict(mapped)
            saved=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
            optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
            if device=='mps':torch.mps.set_rng_state(saved['device_rng'])
            if device=='cuda':torch.cuda.set_rng_state_all(saved['device_rng'])
            start_epoch=state['epoch'];start_microstep=state['next_microstep'];optimizer_step=state['optimizer_step']
            del weights,current,mapped,saved
        write_json(manifest_path,dict(status='training',signature=signature,base_model_fingerprint=data.manifest['base_model_fingerprint'],**({'last_checkpoint':Path(resume).name} if resume else {})))
        log=open(output/'training_log.jsonl','a')
        memory_log=open(output/'memory_log.jsonl','a') if settings['memory_profile'] else None
        try:
            for epoch in range(start_epoch,settings['epochs']):
                order=np.random.default_rng(np.random.SeedSequence([settings['seed'],epoch])).permutation(len(data.pairs));microsteps=math.ceil(len(order)/settings['microbatch'])
                start=start_microstep if epoch==start_epoch else 0;acc=settings['gradient_accumulation'];losses=[]
                if start%acc and start!=microsteps:raise ValueError('Resume is not on an optimizer boundary')
                for micro in range(start,microsteps):
                    window_start=(micro//acc)*acc;window_stop=min(window_start+acc,microsteps)
                    total_examples=min(window_stop*settings['microbatch'],len(order))-window_start*settings['microbatch']
                    indices=order[micro*settings['microbatch']:(micro+1)*settings['microbatch']]
                    batch=data.batch(indices)
                    losses.append(one_backward(model,batch,settings['temperature'],len(indices)/total_examples))
                    if settings['memory_profile'] or settings['mps_cleanup']:
                        record=memory_record(device,settings['mps_cleanup'])
                        if memory_log:
                            record.update(epoch=epoch,microstep=micro+1,optimizer_step=optimizer_step,anchor_lengths=list(map(len,batch['anchors'])),elapsed_seconds=time.perf_counter()-run_started)
                            memory_log.write(json.dumps(record)+'\n');memory_log.flush()
                    if micro+1==window_stop:
                        grad=checked_step(model,optimizer);optimizer_step+=1
                        log.write(json.dumps(dict(epoch=epoch,microstep=micro+1,optimizer_step=optimizer_step,loss=float(np.mean(losses)),gradient_norm=grad))+'\n');log.flush();losses=[]
                        budget_reached=settings['max_run_hours'] is not None and time.perf_counter()-run_started>=settings['max_run_hours']*3600
                        if optimizer_step%settings['checkpoint_every']==0 or micro+1==microsteps or budget_reached:
                            validation_due=optimizer_step%(settings['validation_every'] or settings['checkpoint_every'])==0 or micro+1==microsteps
                            metrics=validation(model,val,settings) if validation_due and not budget_reached else {'skipped':'runtime budget reached' if budget_reached else 'validation not scheduled'}
                            state=dict(signature=signature,epoch=epoch,next_microstep=micro+1,optimizer_step=optimizer_step,validation=metrics)
                            path=save_checkpoint(model,optimizer,output,state)
                            # Keep validation history even after weight checkpoints are pruned.
                            with (output/'validation_log.jsonl').open('a') as validation_log:
                                validation_log.write(json.dumps(dict(epoch=epoch,microstep=micro+1,optimizer_step=optimizer_step,validation=metrics,elapsed_seconds=time.perf_counter()-run_started))+'\n')
                            prune_checkpoints(output,settings['keep_checkpoints'],signature)
                            unfinished=epoch+1<settings['epochs'] or micro+1<microsteps
                            status='budget_exhausted' if budget_reached and unfinished else 'training'
                            write_json(manifest_path,dict(status=status,signature=signature,last_checkpoint=path.name,optimizer_step=optimizer_step,validation=metrics))
                            if status=='budget_exhausted':return
            write_json(manifest_path,dict(status='complete',signature=signature,optimizer_step=optimizer_step,last_checkpoint=path.name))
        except Exception as error:
            previous=json.loads(manifest_path.read_text())
            previous.update(status='failed',error_type=type(error).__name__,error=str(error),last_completed_optimizer_step=optimizer_step)
            write_json(manifest_path,previous)
            raise
        finally:
            log.close()
            if memory_log:memory_log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ['database','fasta','output']:p.add_argument(name)
    p.add_argument('--cds-blocks',help='Restrict positive and negative fragments to individual selected CDS blocks')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-genes',type=int,default=1000);p.add_argument('--validation-genes',type=int,default=100)
    p.add_argument('--pairs-per-gene',type=int,default=200);p.add_argument('--validation-pairs',type=int,default=20)
    p.add_argument('--min-length',type=int,default=50);p.add_argument('--max-length',type=int,default=250)
    p.add_argument('--no-reverse-augment',dest='reverse_augment',action='store_false')
    p=sub.add_parser('pilot')
    for name in ['dataset','output']:p.add_argument(name)
    p.add_argument('--device',default='mps');p.add_argument('--microbatch',type=int,default=2);p.add_argument('--repeats',type=int,default=2)
    p=sub.add_parser('train')
    for name in ['dataset','output']:p.add_argument(name)
    p.add_argument('--device',default='mps');p.add_argument('--config');p.add_argument('--resume')
    args=vars(parser.parse_args());command=args.pop('command');result={'prepare':prepare,'pilot':pilot,'train':train}[command](**args)
    if result is not None:print(json.dumps(result,indent=2))


if __name__=='__main__':main()
