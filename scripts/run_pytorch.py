"""Step 2: measured FP32, true FP16, custom-grid PyTorch PTQ, and native INT8."""
import json,sys,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from software.data import load_idx
from software.model import LeNet,prepare_pixels
from software.torch_quantization import weight_qdq,calibrated_activations,QDQLeNet,convert_native
from software.exports import export_weights,export_floating,export_native
from scripts.train import evaluate

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(7)
    d=ROOT/'results/quantization';d.mkdir(exist_ok=True);start=time.perf_counter()
    state=torch.load(ROOT/'software/lenet_fashion_mnist.pt',map_location='cpu',weights_only=True);model=LeNet().eval();model.load_state_dict(state['state_dict'])
    raw,labels=load_idx(ROOT/'data/fashion','t10k');x=prepare_pixels(raw);y=torch.tensor(labels,dtype=torch.long)
    train,_=load_idx(ROOT/'data/fashion');ci=np.load(ROOT/'results/software/calibration_indices.npy');cx=prepare_pixels(train[ci]);ranges=json.loads((d/'calibration_ranges.json').read_text())
    results=[];payload=61706*4
    base,out=evaluate(model,x,y);results.append({'name':'FP32','method':'float','bits':32,'mode':'floating_point','execution':'native FP32','payload_bytes':payload,'compression_ratio':1,**base})
    half=copy.deepcopy(model).half();half_result,hout=evaluate(half,x.half(),y)
    torch.save({'state_dict':half.state_dict(),'precision':'FP16','author':'Matin Firoozbakht'},ROOT/'models/fp16.pt')
    np.savez_compressed(d/'fp16_predictions.npz',logits=hout,predictions=hout.argmax(1))
    results.append({'name':'FP16','method':'float','bits':16,'mode':'floating_point','execution':'native FP16 CPU','payload_bytes':61706*2,'compression_ratio':2,**half_result})
    export_floating(ROOT/'hardware/exports/fp32',{n:v.detach().numpy() for n,v in model.state_dict().items()},32)
    export_floating(ROOT/'hardware/exports/fp16',{n:v.detach().numpy() for n,v in half.state_dict().items()},16)
    native_export={}
    for bits in [16,8,4,2]:
        for method in ['symmetric','affine']:
            net,wp,codes=weight_qdq(model,bits,method);ap=calibrated_activations(ranges,bits,method)
            weights={n:v.detach().numpy() for n,v in net.state_dict().items()}
            metadata=export_weights(ROOT/f'hardware/exports/{method}_int{bits}',weights,wp,codes,ap,raw[0]);native_export[f'{method}_{bits}']=metadata
            for mode in ['weights_only','weights_activations']:
                a=ap if mode=='weights_activations' else None;qmodel=QDQLeNet(net,a).eval();result,out=evaluate(qmodel,x,y)
                parameter_bytes=metadata['full_qdq_parameter_bytes'] if a else metadata['payload_bytes']
                row={'name':f'INT{bits}','method':method,'bits':bits,'mode':mode,'execution':'FP32 Q/DQ simulation','payload_bytes':parameter_bytes,'compression_ratio':payload/parameter_bytes,**result};results.append(row);print(json.dumps(row),flush=True)
                p=ROOT/f'models/{method}_int{bits}_{mode}.pt'
                torch.save({'state_dict':net.state_dict(),'bits':bits,'method':method,'mode':mode,'activations':{k:v.to_dict() for k,v in a.items()} if a else None,'weight_qparams':{k:v.to_dict() for k,v in wp.items()},'export':f'hardware/exports/{method}_int{bits}','author':'Matin Firoozbakht'},p)
                np.savez_compressed(d/f'torch_{method}_{bits}_{mode}.npz',logits=out,predictions=out.argmax(1))
    native=convert_native(model,[cx[i:i+128] for i in range(0,len(cx),128)])
    result,out=evaluate(native,x,y)
    export_native(ROOT/'hardware/exports/native_int8',native)
    torch.save({'state_dict':native.state_dict(),'engine':torch.backends.quantized.engine,'author':'Matin Firoozbakht'},ROOT/'models/native_int8.pt')
    converted=[]
    for name,module in native.named_modules():
        if isinstance(module,(torch.ao.nn.quantized.Conv2d,torch.ao.nn.quantized.Linear)):
            w=module.weight();converted.append({'layer':name,'operator':type(module).__module__+'.'+type(module).__name__,'weight_dtype':str(w.dtype),'qscheme':str(w.qscheme()),'shape':list(w.shape)})
    assert len(converted)==5
    np.savez_compressed(d/'native_int8_predictions.npz',logits=out,predictions=out.argmax(1))
    native_result={'name':'Native INT8','method':'per_channel_weight_affine_activation','bits':8,'mode':'weights_activations','execution':'quantized CPU operators','checkpoint_bytes':(ROOT/'models/native_int8.pt').stat().st_size,**result}
    result={'status':'completed','torch_version':torch.__version__,'results':results,'native_int8':native_result,'native_operators':converted,'calibration_images':len(cx),'calibration_source':'training split only','seconds':time.perf_counter()-start}
    (d/'pytorch_results.json').write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result),flush=True)
if __name__=='__main__':main()
