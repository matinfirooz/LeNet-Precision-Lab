"""NumPy-only inference directly from packed hardware exports; no neural-network import."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from software.data import CLASSES,load_idx
from software.packed_runtime import PackedLeNet

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bits',type=int,choices=[2,4,8,16],default=8)
    p.add_argument('--method',choices=['symmetric','affine'],default='affine')
    p.add_argument('--mode',choices=['weights_only','weights_activations'],default='weights_only')
    p.add_argument('--index',type=int,default=0,help='official test-image index for single-image inference')
    p.add_argument('--all',action='store_true',help='evaluate all 10,000 official test images')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--output',type=Path,help='optional JSON output path')
    args=p.parse_args()
    if args.batch_size<=0:p.error('--batch-size must be positive')
    raw,labels=load_idx(ROOT/'data/fashion','t10k')
    if not 0<=args.index<len(raw):p.error('--index is outside the test set')
    runtime=PackedLeNet(ROOT/f'hardware/exports/{args.method}_int{args.bits}',args.mode)
    started=time.perf_counter()
    result={'author':'Matin Firoozbakht','github':'https://github.com/matinfirooz','method':args.method,'bits':args.bits,'mode':args.mode,
            'execution':'NumPy FP32 Q/DQ from packed parameter bytes; not an integer-only accelerator','input_sha256':runtime.input_hashes()}
    if args.all:
        measured,logits=runtime.evaluate(raw,labels,args.batch_size);prediction=logits.argmax(1)
        reference=ROOT/f'results/quantization/torch_{args.method}_{args.bits}_{args.mode}.npz'
        result.update(measured)
        if reference.exists():
            saved=np.load(reference);result['saved_torch_prediction_disagreements']=int(np.sum(prediction!=saved['predictions']))
            result['saved_torch_max_logit_error']=float(np.max(abs(logits-saved['logits'])))
        result['per_class']=[{'class':CLASSES[c],'n':int(np.sum(labels==c)),'correct':int(np.sum((labels==c)&(prediction==c)))} for c in range(10)]
    else:
        probability=runtime.probabilities(raw[args.index:args.index+1])[0];pred=int(np.argmax(probability))
        result.update(index=args.index,prediction=pred,class_name=CLASSES[pred],true_class=int(labels[args.index]),probabilities=probability.tolist())
    result['seconds']=time.perf_counter()-started
    result['neural_network_imports']=[n for n in sys.modules if n.split('.')[0] in ['torch','tensorflow','jax','keras']]
    assert not result['neural_network_imports']
    text=json.dumps(result,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+'\n')
    print(text)
if __name__=='__main__':main()
