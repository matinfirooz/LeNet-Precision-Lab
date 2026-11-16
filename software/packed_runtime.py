"""Run the packed hardware parameter image through NumPy Q/DQ inference."""
import hashlib
from pathlib import Path
import numpy as np
from .exports import load_export,load_activation_params
from .numpy_lenet import evaluate,forward,prepare_pixels

class PackedLeNet:
    """Reads binary weights, biases and grids; does not require a PT or NPZ checkpoint."""
    def __init__(self,export,mode='weights_only'):
        if mode not in ('weights_only','weights_activations'):raise ValueError('unknown activation policy')
        self.root=Path(export);self.mode=mode
        self.weights,self.metadata=load_export(self.root)
        self.activations=load_activation_params(self.root,self.metadata) if mode=='weights_activations' else None
    def logits(self,raw):
        raw=np.asarray(raw)
        if raw.ndim!=3 or raw.shape[1:]!=(28,28) or not len(raw):raise ValueError('images must have shape N,28,28')
        if not np.isfinite(raw).all() or np.any(raw<0) or np.any(raw>255):raise ValueError('pixels must be finite in 0..255')
        return forward(prepare_pixels(raw),self.weights,self.activations)
    def probabilities(self,raw):
        scores=self.logits(raw);scores=scores-scores.max(1,keepdims=True)
        exp=np.exp(scores);return exp/exp.sum(1,keepdims=True)
    def evaluate(self,raw,labels,batch=128):
        return evaluate(raw,labels,self.weights,self.activations,batch=batch)
    def input_hashes(self):
        names=['weights.bin','biases_fp32.bin','qparams.bin','metadata.json']
        if self.activations is not None:names.append('activation_qparams.bin')
        return {n:hashlib.sha256((self.root/n).read_bytes()).hexdigest() for n in names}
