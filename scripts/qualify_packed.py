"""Evaluate every packed custom-grid model on the full official test set, NumPy only."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from software.data import load_idx
from software.packed_runtime import PackedLeNet
import numpy as np

def main():
    raw,labels=load_idx(ROOT/'data/fashion','t10k');rows=[];started=time.perf_counter()
    for bits in [16,8,4,2]:
        for method in ['symmetric','affine']:
            for mode in ['weights_only','weights_activations']:
                model=PackedLeNet(ROOT/f'hardware/exports/{method}_int{bits}',mode)
                measured,logits=model.evaluate(raw,labels)
                ref=np.load(ROOT/f'results/quantization/torch_{method}_{bits}_{mode}.npz')
                row={'bits':bits,'method':method,'mode':mode,**measured,
                     'saved_torch_prediction_disagreements':int(np.sum(logits.argmax(1)!=ref['predictions'])),
                     'saved_torch_max_logit_error':float(np.max(abs(logits-ref['logits']))),'input_sha256':model.input_hashes()}
                rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='input_sha256'}),flush=True)
    result={'status':'completed','author':'Matin Firoozbakht','github':'https://github.com/matinfirooz','execution':'NumPy FP32 Q/DQ from decoded hardware parameter binaries','results':rows,'seconds':time.perf_counter()-started,'neural_network_imports':[n for n in sys.modules if n.split('.')[0] in ['torch','tensorflow','jax','keras']]}
    assert not result['neural_network_imports']
    (ROOT/'results/quantization/packed_results.json').write_text(json.dumps(result,indent=2));print('Saved 16 complete packed-model evaluations.')
if __name__=='__main__':main()
