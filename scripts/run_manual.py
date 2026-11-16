"""Step 1 runs in a separate NumPy-only process; never imports a neural network library."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from software.data import load_idx,split_indices
from software.quantization import SAMPLE,parameters,quantize,dequantize
from software.numpy_lenet import calibrate,activation_parameters,quantized_weights,evaluate

def main():
    start=time.perf_counter();r=ROOT/'results/quantization';r.mkdir(parents=True,exist_ok=True)
    weights=dict(np.load(ROOT/'software/trained_weights.npz'));raw,labels=load_idx(ROOT/'data/fashion','t10k')
    train,ty=load_idx(ROOT/'data/fashion');ti,vi=split_indices(ty)
    rng=np.random.default_rng(101);ci=np.sort(rng.choice(ti,1000,replace=False))
    np.save(ROOT/'results/software/calibration_indices.npy',ci)
    ranges=calibrate(train[ci],weights);(r/'calibration_ranges.json').write_text(json.dumps(ranges,indent=2))
    sample=[]
    for method in ['symmetric','endpoint','affine']:
        for bits in [16,8,4]:
            p=parameters(SAMPLE,bits,method);q=quantize(SAMPLE,p);dq=dequantize(q,p,dtype=np.float64)
            sample.append({'method':method,'bits':bits,**p.to_dict(),'q':q.tolist(),'dequantized':dq.tolist(),'mse':float(np.mean((dq-SAMPLE)**2)),'max_error':float(np.max(abs(dq-SAMPLE)))})
    (r/'worked_example.json').write_text(json.dumps(sample,indent=2))
    baseline,logits=evaluate(raw,labels,weights)
    ref=np.load(ROOT/'results/software/float_test_outputs.npz')['logits']
    baseline['torch_max_logit_error']=float(np.max(abs(logits-ref)));baseline['torch_prediction_disagreements']=int(np.sum(logits.argmax(1)!=ref.argmax(1)))
    np.savez_compressed(r/'numpy_fp32_predictions.npz',logits=logits,labels=labels)
    results=[]
    for bits in [16,8,4]:
        for method in ['symmetric','affine','endpoint']:
            qw,wp=quantized_weights(weights,bits,method)
            for mode in ['weights_only','weights_activations']:
                ap=activation_parameters(ranges,bits,method) if mode=='weights_activations' else None
                result,out=evaluate(raw,labels,qw,ap)
                row={'method':method,'bits':bits,'mode':mode,**result};results.append(row);print(json.dumps(row),flush=True)
                np.savez_compressed(r/f'numpy_{method}_{bits}_{mode}.npz',logits=out,predictions=out.argmax(1))
    result={'status':'completed','baseline':baseline,'results':results,'seconds':time.perf_counter()-start,'calibration_images':1000,'calibration_seed':101,'numpy_version':np.__version__,'neural_network_imports':[n for n in sys.modules if n.split('.')[0] in ['torch','tensorflow','jax','keras']]}
    assert not result['neural_network_imports'];assert baseline['torch_prediction_disagreements']==0
    (r/'manual_results.json').write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result),flush=True)
if __name__=='__main__':main()
