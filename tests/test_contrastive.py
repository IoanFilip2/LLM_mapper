import copy,json
import numpy as np
import pytest
import torch
from llm_mapper.finetuning.data import sample_triplet
from llm_mapper.finetuning.model import info_nce,negative_mask


def test_triplets_are_correct_reproducible_and_never_self_negative():
    seq=['ACGT'*100,'AAAA'*100,'GGCC'*100];a=np.random.default_rng(5);b=np.random.default_rng(5)
    for _ in range(30):
        row=sample_triplet(a,0,[0,1,2],seq);assert row==sample_triplet(b,0,[0,1,2],seq)
        anchor,start,length,donor,ns,nl,pr,nr=row
        assert 50<=length<=250 and 50<=nl<=250 and donor!=anchor
        assert start+length<=len(seq[anchor]) and ns+nl<=len(seq[donor])
        assert seq[donor][ns:ns+nl] not in seq[anchor]


def test_infonce_has_correct_probability_and_gradients_for_both_branches():
    a=torch.tensor([[1.,0.]],requires_grad=True);f=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
    loss=info_nce(a,f,torch.ones((1,2),dtype=torch.bool),temperature=1.)
    assert loss.item()==pytest.approx(np.log1p(np.exp(-1)))
    loss.backward();assert a.grad.abs().sum()>0 and f.grad.abs().sum()>0
    with pytest.raises(ValueError):info_nce(a,f,torch.tensor([[True,False]]))


def test_false_negative_mask_handles_same_gene_and_reverse_complement_match():
    batch=dict(anchors=['AAAACCCC','AAAACCCC'],positives=['AAAA','CCCC'],negatives=['TTTT','AGAG'],anchor_ids=[0,0],negative_ids=[2,3])
    mask=negative_mask(batch)
    assert mask.tolist()==[[True,False,False,True],[False,True,False,True]]


def test_sdpa_training_gradients_match_eager_and_export_reload(tmp_path):
    pytest.importorskip('modelgenerator')
    from modelgenerator.huggingface_models.rnabert.configuration_rnabert import RNABertConfig
    from modelgenerator.huggingface_models.rnabert.modeling_rnabert import RNABertModel
    from modelgenerator.huggingface_models.rnabert.tokenization_rnabert import RNABertTokenizer
    from llm_mapper.embedders import EmbeddingConfig,create_embedder
    from llm_mapper.finetuning.model import ContrastiveEncoder
    vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','[BOS]','[EOS]','A','C','G','T','N','U']
    (tmp_path/'vocab.txt').write_text('\n'.join(vocab)+'\n');RNABertTokenizer(str(tmp_path/'vocab.txt')).save_pretrained(tmp_path)
    torch.manual_seed(9)
    config=RNABertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32,max_position_embeddings=128,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
    RNABertModel(config).save_pretrained(tmp_path)
    batch=dict(anchors=['ACGT'*20,'GGGA'*10],positives=['ACGTACGT','GGGAGGGA'],negatives=['GGGAGGGA','ACGTACGT'],anchor_ids=[0,1],negative_ids=[1,0])
    models=[ContrastiveEncoder(EmbeddingConfig(model=str(tmp_path),max_length=128,attention=att),'cpu',chunk_length=32,checkpointing=True) for att in ['eager','sdpa']]
    models.append(ContrastiveEncoder(EmbeddingConfig(model=str(tmp_path),max_length=128,attention='sdpa'),'cpu',chunk_length=32,checkpointing=False))
    gradients=[]
    for model in models:
        model.train();anchor,fragment=model(batch);loss=info_nce(anchor,fragment,negative_mask(batch));loss.backward()
        gradients.append({k.replace('.original.','.'):p.grad.detach().clone() for k,p in model.model.named_parameters() if p.grad is not None})
    assert gradients[0].keys()==gradients[1].keys()
    for name in gradients[0]:torch.testing.assert_close(gradients[0][name],gradients[1][name],atol=2e-5,rtol=2e-3)
    for name in gradients[1]:torch.testing.assert_close(gradients[1][name],gradients[2][name],atol=2e-5,rtol=2e-3)
    sdpa=models[1];sdpa.eval();before=sdpa.encode_raw(['ACGT'*8]).detach().numpy();out=tmp_path/'finetuned';sdpa.save_encoder(out)
    loaded=create_embedder(EmbeddingConfig(model=str(out),max_length=128,attention='sdpa'),'cpu')
    np.testing.assert_allclose(before,loaded.encode(['ACGT'*8]),atol=1e-6)
    # Multi-context differentiable pooling agrees with the reference recipe.
    with torch.no_grad():
        chunks=['ACGT'*8,'ACGT'*8,'ACGT'*4]
        raw=[sdpa.encode_raw([s])[0] for s in chunks]
        expected=sum(v*len(s) for s,v in zip(chunks,raw))/80
        expected=torch.nn.functional.normalize(expected,dim=0)
        torch.testing.assert_close(sdpa.encode_genes(['ACGT'*20])[0],expected)


def test_training_checkpoint_resume_matches_uninterrupted_tiny_run(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from safetensors.torch import save_file,load_file
    from llm_mapper.embedders import EmbeddingConfig
    from llm_mapper.finetuning import runner
    class Data:
        def __init__(self,*args):
            self.signature='tiny-dataset';self.manifest={'base_model_fingerprint':'tiny'}
            self.pairs=np.array([[0],[1],[0],[1],[0],[1]])
        def batch(self,indices):
            ids=[int(self.pairs[i,0]) for i in indices];seq=['AAAACC','GGGGTT'];pos=['AAA','GGG']
            return dict(anchors=[seq[i] for i in ids],positives=[pos[i] for i in ids],negatives=[pos[1-i] for i in ids],anchor_ids=ids,negative_ids=[1-i for i in ids])
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=torch.nn.Linear(4,3,bias=False);self.device='cpu'
            self.adapter=SimpleNamespace(config=EmbeddingConfig(),fingerprint='tiny')
        def forward(self,batch):
            def encode(strings):
                x=torch.tensor([[s.count(c)/len(s) for c in 'ACGT'] for s in strings],dtype=torch.float32)
                return torch.nn.functional.normalize(self.model(x),dim=-1)
            return encode(batch['anchors']),encode(batch['positives']+batch['negatives'])
        def save_encoder(self,path):save_file(self.model.state_dict(),str(path/'model.safetensors'))
    def setup(data,device,settings):
        runner.seed_everything(settings['seed'],device);model=Toy()
        return model,torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],foreach=False)
    monkeypatch.setattr(runner,'TripletData',Data);monkeypatch.setattr(runner,'setup',setup)
    config=tmp_path/'settings.json';config.write_text(json.dumps(dict(microbatch=1,gradient_accumulation=2,epochs=2,checkpoint_every=1,keep_checkpoints=2)))
    baseline=tmp_path/'baseline';resumed=tmp_path/'resumed';runner.train('tiny',baseline,device='cpu',config=config)
    original=runner.save_checkpoint
    def interrupt(*args):
        path=original(*args);raise RuntimeError('simulated interruption')
    monkeypatch.setattr(runner,'save_checkpoint',interrupt)
    with pytest.raises(RuntimeError,match='simulated interruption'):runner.train('tiny',resumed,device='cpu',config=config)
    monkeypatch.setattr(runner,'save_checkpoint',original)
    runner.train('tiny',resumed,device='cpu',config=config,resume=resumed/'checkpoint-000001')
    for key,value in load_file(str(baseline/'checkpoint-000006/model.safetensors')).items():
        torch.testing.assert_close(value,load_file(str(resumed/'checkpoint-000006/model.safetensors'))[key],rtol=0,atol=0)
    assert len(list(resumed.glob('checkpoint-*')))==2
    # Resuming an already finished checkpoint is also well-defined.
    runner.train('tiny',resumed,device='cpu',config=config,resume=resumed/'checkpoint-000006')
    assert json.loads((resumed/'training_manifest.json').read_text())['status']=='complete'
    budget_config=tmp_path/'budget.json';budget_config.write_text(json.dumps(dict(microbatch=1,gradient_accumulation=2,epochs=1,checkpoint_every=100,keep_checkpoints=2,max_run_hours=1e-9)))
    budget=tmp_path/'budget_run';runner.train('tiny',budget,device='cpu',config=budget_config)
    status=json.loads((budget/'training_manifest.json').read_text())
    assert status['status']=='budget_exhausted' and status['optimizer_step']==1
    runner.train('tiny',budget,device='cpu',config=budget_config,resume=budget/status['last_checkpoint'])
    status=json.loads((budget/'training_manifest.json').read_text());assert status['status']=='budget_exhausted'
    runner.train('tiny',budget,device='cpu',config=budget_config,resume=budget/status['last_checkpoint'])
    assert json.loads((budget/'training_manifest.json').read_text())['status']=='complete'


def test_exon_sampling_never_crosses_junctions_and_supports_both_ends():
    from llm_mapper.finetuning.data import sample_fragment
    rng=np.random.default_rng(10);seen=set()
    for _ in range(1000):
        start,length=sample_fragment(rng,'A'*140,50,250,[(0,60),(60,140)])
        assert 50<=length<=80
        assert any(a<=start and start+length<=b for a,b in [(0,60),(60,140)])
        seen.add(start)
    assert 0 in seen and 60 in seen
    # A one-base number of valid starts is allowed, including the block end.
    assert sample_fragment(rng,'A'*100,50,50,[(50,100)])==(50,50)
    sequences=['ACGT'*100,'GGCC'*100]
    regions=[[(0,70),(70,400)],[(0,90),(90,400)]]
    for _ in range(50):
        a,ps,pl,n,ns,nl,*_=sample_triplet(rng,0,[0,1],sequences,regions=regions)
        assert any(x<=ps and ps+pl<=y for x,y in regions[a])
        assert any(x<=ns and ns+nl<=y for x,y in regions[n])


def test_prepare_exon_dataset_validates_blocks_and_splits(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from llm_mapper.finetuning import data as module
    from llm_mapper.preparation.downloads import file_hash
    import pysam
    sequences={'A':'ACGT'*30,'B':'GGCC'*30,'C':'ATAT'*30,'D':'CCCA'*30}
    fasta=tmp_path/'genes.fa';fasta.write_text(''.join(f'>{g}\n{s}\n' for g,s in sequences.items()));pysam.faidx(str(fasta))
    genes=[dict(gene_symbol=g,sequence_length=len(s),transcript_id=g) for g,s in sequences.items()]
    db=SimpleNamespace(genes=genes,manifest={'signature':dict(fasta_sha256=file_hash(fasta),model_config={},model_fingerprint='tiny',chunk_length=100)})
    monkeypatch.setattr(module,'GeneVectorDatabase',lambda _:db)
    blocks=tmp_path/'blocks.jsonl';blocks.write_text(''.join(json.dumps(dict(gene_symbol=g,transcript_id=g,cds_start=a,cds_end=b,genomic_start=a+1000,genomic_end=b+1000))+'\n' for g in sequences for a,b in [(0,60),(60,120)]))
    output=tmp_path/'data';m=module.prepare('db',fasta,output,train_genes=2,validation_genes=2,pairs_per_gene=4,validation_pairs=4,cds_blocks=blocks)
    assert m['fragment_scope']=='single_CDS_block'
    d=module.TripletData(output);v=module.TripletData(output,'validation')
    assert set(d.pairs[:,0]).isdisjoint(set(v.pairs[:,0]))
    for split in [d,v]:
        for a,ps,pl,n,ns,nl,*_ in split.pairs:
            assert any(x<=ps and ps+pl<=y for x,y in split.genes[a]['cds_blocks'])
            assert any(x<=ns and ns+nl<=y for x,y in split.genes[n]['cds_blocks'])


def test_resume_allows_only_operational_setting_changes():
    from llm_mapper.finetuning.runner import compatible_signature
    a={'dataset_sha256':'abc','device':'mps','settings':{'microbatch':2,'checkpoint_every':250}}
    b=copy.deepcopy(a);b['settings'].update(checkpoint_every=25,memory_profile=True,mps_cleanup=True,validation_every=250)
    assert compatible_signature(a,b)
    b['settings']['microbatch']=1
    assert not compatible_signature(a,b)


def test_memory_profile_separates_gc_and_cache_release(monkeypatch):
    from llm_mapper.finetuning import runner
    calls=[]
    monkeypatch.setattr(runner.gc,'collect',lambda: calls.append('gc') or 7)
    monkeypatch.setattr(torch.mps,'synchronize',lambda: None)
    monkeypatch.setattr(torch.mps,'current_allocated_memory',lambda: 100)
    monkeypatch.setattr(torch.mps,'driver_allocated_memory',lambda: 200)
    monkeypatch.setattr(torch.mps,'empty_cache',lambda: calls.append('cache'))
    result=runner.memory_record('mps',True)
    assert calls==['gc','cache'] and result['gc_collected']==7
    for stage in ['before','after_gc','after_cache_clear']:
        assert result[stage]['tensor_bytes']==100 and result[stage]['driver_bytes']==200
