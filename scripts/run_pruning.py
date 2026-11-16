"""Additional pruning study: fixed rates, mask-preserving recovery and channel compaction."""
import json,sys,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from software.model import LeNet,prepare_pixels
from software.data import load_idx,split_indices
from software.pruning import global_prune,remove_masks,counts,sparse_export,compact_channels,dense_macs
from software.torch_quantization import weight_qdq,QDQLeNet
from scripts.train import evaluate

def recover(model,x,y,vx,vy,path,epochs=3):
    # Masks remain attached during optimization, so pruned weights cannot regrow.
    torch.manual_seed(17);rng=torch.Generator().manual_seed(17)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    best=float('inf');best_epoch=0;history=[];best_state=None
    for epoch in range(1,epochs+1):
        model.train();order=torch.randperm(len(y),generator=rng)
        for begin in range(0,len(y),256):
            i=order[begin:begin+256];optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(x[i]),y[i]);loss.backward();optimizer.step()
        v,_=evaluate(model,vx,vy);history.append({'epoch':epoch,**v});print('RECOVER',path.name,json.dumps(history[-1]),flush=True)
        if v['cross_entropy']<best:best=v['cross_entropy'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state);model.eval()
    return model,{'epochs':epochs,'selected_epoch':best_epoch,'history':history,'selection':'validation cross-entropy only; checkpoint among three recovery epochs','learning_rate':.0001}

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(7);torch.use_deterministic_algorithms(True);start=time.perf_counter()
    model=LeNet();model.load_state_dict(torch.load(ROOT/'software/lenet_fashion_mnist.pt',map_location='cpu',weights_only=True)['state_dict']);model.eval()
    raw,lab=load_idx(ROOT/'data/fashion','t10k');tx=prepare_pixels(raw);ty=torch.tensor(lab,dtype=torch.long)
    tr,labels=load_idx(ROOT/'data/fashion');ti,vi=split_indices(labels);allx=prepare_pixels(tr);ally=torch.tensor(labels,dtype=torch.long)
    x,y=allx[ti],ally[ti];vx,vy=allx[vi],ally[vi];results=[]
    for rate in [0,.25,.5,.75,.9]:
        m=global_prune(model,rate);result,out=evaluate(m,tx,ty);stat=counts(m);remove_masks(m)
        export=sparse_export(m,ROOT/f'hardware/exports/pruned_{int(rate*100)}_csr')
        torch.save({'state_dict':m.state_dict(),'rate':rate,'author':'Matin Firoozbakht'},ROOT/f'models/pruned_{int(rate*100)}.pt')
        row={'rate':rate,'kind':'global magnitude, no recovery',**stat,**result,'sparse_payload_bytes':export['payload_bytes'],'sparse_compression_ratio':export['compression_ratio'],'dense_macs':dense_macs()};results.append(row);print(json.dumps(row),flush=True)
        np.savez_compressed(ROOT/f'results/pruning/pruned_{int(rate*100)}_predictions.npz',predictions=out.argmax(1))
    m=global_prune(model,.75);m,history=recover(m,x,y,vx,vy,ROOT/'models/pruned_75_recovered.pt')
    result,out=evaluate(m,tx,ty);stat=counts(m);remove_masks(m)
    export=sparse_export(m,ROOT/'hardware/exports/pruned_75_recovered_csr')
    torch.save({'state_dict':m.state_dict(),'rate':.75,'history':history,'author':'Matin Firoozbakht'},ROOT/'models/pruned_75_recovered.pt')
    recovered={'kind':'75% global magnitude with recovery',**stat,**result,'history':history,'sparse_payload_bytes':export['payload_bytes'],'dense_macs':dense_macs()}
    q,_,_=weight_qdq(m,8,'symmetric');joint_result,_=evaluate(QDQLeNet(q).eval(),tx,ty)
    torch.save({'state_dict':q.state_dict(),'bits':8,'method':'symmetric','mode':'weights_only','activations':None,'author':'Matin Firoozbakht'},ROOT/'models/pruned_75_recovered_int8_qdq.pt')
    compact,selected,masked=compact_channels(model)
    with torch.inference_mode():error=(compact(tx[:64])-masked(tx[:64])).abs().max().item()
    assert error<1e-5,error
    before,_=evaluate(compact,tx,ty);stat=counts(compact)
    compact,history=recover(compact,x,y,vx,vy,ROOT/'models/compact_recovered.pt')
    after,out=evaluate(compact,tx,ty)
    torch.save({'state_dict':compact.state_dict(),'channels':[4,12,90,63],'selected_channels':selected,'history':history,'author':'Matin Firoozbakht'},ROOT/'models/compact_recovered.pt')
    np.savez_compressed(ROOT/'models/compact_weights.npz',**{k:v.detach().numpy() for k,v in compact.state_dict().items()})
    np.savez_compressed(ROOT/'results/pruning/compact_predictions.npz',predictions=out.argmax(1))
    result={'status':'completed','scope':'controlled extension: pruning rates are not prescribed in the assignment','rates':results,'recovered':recovered,'joint_weight_int8':joint_result,
      'compact':{'channels':[4,12,90,63],**stat,'dense_macs':dense_macs((4,12,90,63)),'dense_fp32_bytes':stat['parameters']*4,'before_recovery':before,'after_recovery':after,'history':history,'masked_equivalence_max_error':error,'selected_channels':selected},'seconds':time.perf_counter()-start}
    (ROOT/'results/pruning/pruning_results.json').write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result),flush=True)
if __name__=='__main__':main()
