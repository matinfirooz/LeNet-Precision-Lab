"""Global magnitude masks, CSR weight exports and actual channel compaction."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import prune
from .model import LeNet
from .numpy_lenet import LAYERS

class CompactLeNet(LeNet):
    def __init__(self,channels=(4,12,90,63)):
        torch.nn.Module.__init__(self);a,b,c,d=channels
        self.conv1=torch.nn.Conv2d(1,a,5);self.conv3=torch.nn.Conv2d(a,b,5);self.conv5=torch.nn.Conv2d(b,c,5)
        self.fc6=torch.nn.Linear(c,d);self.fc7=torch.nn.Linear(d,10)

def global_prune(model,amount):
    model=copy.deepcopy(model)
    prune.global_unstructured([(getattr(model,n),'weight') for n in LAYERS],pruning_method=prune.L1Unstructured,amount=amount)
    return model

def remove_masks(model):
    for n in LAYERS:
        m=getattr(model,n)
        if hasattr(m,'weight_orig'):prune.remove(m,'weight')
    return model

def counts(model):
    weights=[getattr(model,n).weight.detach() for n in LAYERS]
    total=sum(v.numel() for v in weights);zero=sum((v==0).sum().item() for v in weights)
    return {'parameters':sum(p.numel() for p in model.parameters()),'weight_count':total,'zero_weights':zero,'weight_sparsity':zero/total}

def sparse_export(model,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);rows=[];size=0
    for name in LAYERS:
        m=getattr(model,name);original=m.weight.detach().numpy();a=original.reshape(original.shape[0],-1)
        rr,cc=np.nonzero(a);value=a[rr,cc].astype('<f4');column=cc.astype('<u2');pointer=np.concatenate([[0],np.cumsum(np.bincount(rr,minlength=len(a)))]).astype('<u4')
        bias=m.bias.detach().numpy().astype('<f4');arrays={'values':value,'columns':column,'rowptr':pointer,'bias':bias}
        restored=np.zeros_like(a)
        for i in range(len(a)):restored[i,column[pointer[i]:pointer[i+1]]]=value[pointer[i]:pointer[i+1]]
        assert np.array_equal(restored,a)
        for part,arr in arrays.items():(root/f'{name}_{part}.bin').write_bytes(arr.tobytes())
        layer_bytes=sum(a.nbytes for a in arrays.values());size+=layer_bytes
        rows.append({'layer':name,'shape':list(original.shape),'nnz':len(value),'bytes':layer_bytes,'column_type':'uint16 little-endian','row_pointer_type':'uint32 little-endian'})
    result={'format':'CSR per output channel; convolution input dimension flattened C,H,W','value_type':'FP32 little-endian','payload_bytes':size,'dense_fp32_bytes':61706*4,'compression_ratio':61706*4/size,'layers':rows}
    (root/'metadata.json').write_text(json.dumps(result,indent=2));return result

def compact_channels(model,channels=(4,12,90,63)):
    """Rank original output weight rows by L1; propagate retained input indices."""
    selected={}
    for name,keep in zip(LAYERS[:4],channels):
        w=getattr(model,name).weight.detach();score=w.abs().reshape(len(w),-1).sum(1)
        selected[name]=torch.topk(score,keep,sorted=False).indices.sort().values
    compact=CompactLeNet(channels);previous=torch.arange(1)
    masked=copy.deepcopy(model)
    with torch.no_grad():
        for name in LAYERS:
            source=getattr(model,name);dest=getattr(compact,name)
            keep=selected[name] if name in selected else torch.arange(10)
            dest.weight.copy_(source.weight[keep][:,previous]);dest.bias.copy_(source.bias[keep])
            if name in selected:
                drop=torch.ones(source.weight.shape[0],dtype=torch.bool);drop[keep]=False
                getattr(masked,name).weight[drop]=0;getattr(masked,name).bias[drop]=0
            previous=keep
    return compact.eval(),{n:v.tolist() for n,v in selected.items()},masked.eval()

def dense_macs(channels=(6,16,120,84)):
    a,b,c,d=channels
    return 28*28*a*25+10*10*b*a*25+c*b*25+d*c+10*d
