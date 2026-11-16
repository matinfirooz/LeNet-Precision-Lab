"""LeNet inference using NumPy only, including optional calibrated activation Q/DQ."""
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .quantization import parameters,qdq

LAYERS=('conv1','conv3','conv5','fc6','fc7')
POINTS=('input','t1','p1','t3','p3','t5','t6','logits')
INPUT_POINT={'conv1':'input','conv3':'p1','conv5':'p3','fc6':'t5','fc7':'t6'}

def prepare_pixels(raw):return np.pad(np.asarray(raw,dtype=np.float32)[:,None]/255,((0,0),(0,0),(2,2),(2,2)))

def convolution(x,w,b):
    n,c,h,ww=x.shape;o,_,kh,kw=w.shape
    v=sliding_window_view(x,(kh,kw),axis=(2,3))
    col=np.ascontiguousarray(v.transpose(0,2,3,1,4,5)).reshape(-1,c*kh*kw)
    out=col@w.reshape(o,-1).T+b
    return out.reshape(n,h-kh+1,ww-kw+1,o).transpose(0,3,1,2)

def pool(x):
    n,c,h,w=x.shape
    return x.reshape(n,c,h//2,2,w//2,2).mean((3,5),dtype=np.float32)

def forward(x,weights,activations=None,observe=None):
    def at(name,value):
        if observe is not None:observe(name,value)
        return qdq(value,activations[name]) if activations is not None else value
    def conv(name,x):return convolution(x,weights[name+'.weight'],weights[name+'.bias'])
    def dense(name,x):return x@weights[name+'.weight'].T+weights[name+'.bias']
    x=at('input',x);x=at('t1',np.tanh(conv('conv1',x)))
    x=at('p1',np.tanh(pool(x)));x=at('t3',np.tanh(conv('conv3',x)))
    x=at('p3',np.tanh(pool(x)));x=at('t5',np.tanh(conv('conv5',x)).reshape(len(x),-1))
    x=at('t6',np.tanh(dense('fc6',x)));return at('logits',dense('fc7',x))

def calibrate(raw,weights,batch=128):
    ranges={n:[float('inf'),-float('inf')] for n in POINTS}
    def observe(name,x):ranges[name][0]=min(ranges[name][0],float(x.min()));ranges[name][1]=max(ranges[name][1],float(x.max()))
    for begin in range(0,len(raw),batch):forward(prepare_pixels(raw[begin:begin+batch]),weights,observe=observe)
    return ranges

def activation_parameters(ranges,bits,method):return {n:parameters(v,bits,method) for n,v in ranges.items()}

def quantized_weights(weights,bits,method):
    params={};out={}
    for name,value in weights.items():
        if name.endswith('.weight'):params[name]=parameters(value,bits,method);out[name]=qdq(value,params[name])
        else:out[name]=value.astype(np.float32).copy()
    return out,params

def evaluate(raw,labels,weights,activations=None,batch=128):
    parts=[]
    for begin in range(0,len(raw),batch):parts.append(forward(prepare_pixels(raw[begin:begin+batch]),weights,activations))
    logits=np.concatenate(parts);pred=logits.argmax(1);shift=logits-logits.max(1,keepdims=True)
    loss=float(np.mean(np.log(np.exp(shift).sum(1))-shift[np.arange(len(labels)),labels]))
    return {'n':len(labels),'correct':int(np.sum(pred==labels)),'accuracy':float(np.mean(pred==labels)),'cross_entropy':loss},logits
