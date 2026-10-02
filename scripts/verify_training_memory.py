"""Bounded recovery probe; checkpoint weights are loaded but never overwritten."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from llm_mapper.finetuning.data import TripletData
from llm_mapper.finetuning.runner import DEFAULTS,setup,one_backward,checked_step
from llm_mapper.preparation.downloads import file_hash,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);args=p.parse_args();root=args.root
    checkpoint=root/'training/checkpoint-000250'
    manifest=json.loads((checkpoint/'checkpoint_manifest.json').read_text())
    for filename,key in [('model.safetensors','weight_sha256'),('training_state.pt','training_state_sha256')]:
        assert file_hash(checkpoint/filename)==manifest[key]
    settings={**DEFAULTS,**manifest['state']['signature']['settings']}
    data=TripletData(root/'data');model,optimizer=setup(data,'mps',settings)
    weights=load_file(str(checkpoint/'model.safetensors'))
    model.model.load_state_dict({k:weights[k.replace('.attention.self.original.','.attention.self.')] for k in model.model.state_dict()})
    del weights
    saved=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(saved['optimizer']);del saved
    order=np.random.default_rng(np.random.SeedSequence([settings['seed'],0])).permutation(len(data.pairs))
    records=[];started=time.perf_counter()
    # Reproduce the entire accumulation window immediately following update 469.
    for micro in range(3752,3760):
        indices=order[micro*2:(micro+1)*2];batch=data.batch(indices)
        loss=one_backward(model,batch,settings['temperature'],1/8)
        torch.mps.synchronize()
        records.append(dict(microstep=micro+1,anchor_lengths=list(map(len,batch['anchors'])),loss=loss,driver_bytes=torch.mps.driver_allocated_memory()))
        print(json.dumps(records[-1]),flush=True)
    checked_step(model,optimizer)
    longest=max(range(len(data.pairs)),key=lambda i:len(data.genes[int(data.pairs[i,0])]['sequence']))
    batch=data.batch([longest,longest]);loss=one_backward(model,batch,settings['temperature']);norm=checked_step(model,optimizer)
    torch.mps.synchronize()
    report=dict(status='passed',context_checkpointing=True,weights_discarded=True,records=records,stress_anchor_lengths=list(map(len,batch['anchors'])),stress_loss=loss,stress_gradient_norm=norm,driver_bytes=torch.mps.driver_allocated_memory(),elapsed_seconds=time.perf_counter()-started)
    write_json(root/'memory_recovery_probe.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
