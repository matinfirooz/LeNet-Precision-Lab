"""Single-image CPU latency, one thread, warmed in a quiet process; no Q/DQ speed claims."""
import json,sys,time,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from software.model import LeNet,prepare_pixels
from software.pruning import CompactLeNet,dense_macs
from software.torch_quantization import convert_native
from software.data import load_idx

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    d=torch.load(ROOT/'software/lenet_fashion_mnist.pt',weights_only=True);base=LeNet().eval();base.load_state_dict(d['state_dict'])
    half=LeNet().half().eval();half.load_state_dict(torch.load(ROOT/'models/fp16.pt',weights_only=True)['state_dict'])
    pruned=LeNet().eval();pruned.load_state_dict(torch.load(ROOT/'models/pruned_75_recovered.pt',weights_only=True)['state_dict'])
    compact=CompactLeNet().eval();compact.load_state_dict(torch.load(ROOT/'models/compact_recovered.pt',weights_only=True)['state_dict'])
    native=convert_native(base,[torch.zeros(1,1,32,32)])
    native.load_state_dict(torch.load(ROOT/'models/native_int8.pt',weights_only=True)['state_dict'])
    raw,labels=load_idx(ROOT/'data/fashion','t10k');x=prepare_pixels(raw[:1]);results=[]
    with torch.inference_mode():
        for name,m,xx,macs in [('FP32',base,x,dense_macs()),('FP16',half,x.half(),dense_macs()),('Native INT8',native,x,dense_macs()),('75% sparse weights, dense kernels',pruned,x,dense_macs()),('Compacted FP32',compact,x,dense_macs((4,12,90,63)))]:
            for _ in range(100):m(xx)
            values=[]
            for _ in range(500):
                t=time.perf_counter_ns();m(xx);values.append((time.perf_counter_ns()-t)/1000)
            results.append({'model':name,'batch_size':1,'median_us':float(np.median(values)),'p90_us':float(np.percentile(values,90)),'minimum_us':float(np.min(values)),'dense_macs':macs})
    cpu='unknown'
    p=Path('/proc/cpuinfo')
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith('model name'):cpu=line.split(':',1)[1].strip();break
    result={'status':'completed','cpu':cpu,'threads':1,'warmup_calls':100,'measured_calls':500,'torch_version':torch.__version__,'engine':torch.backends.quantized.engine,'platform':platform.platform(),'results':results,'limitations':'Shared CPU, one batch size, Python dispatch included; no energy or FPGA timing measurements.'}
    (ROOT/'results/software/latency.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
