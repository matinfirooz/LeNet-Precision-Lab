"""Manual quantization. This module and its dependencies never import PyTorch."""
from dataclasses import dataclass, asdict
import math
import numpy as np

METHODS=('symmetric','affine','endpoint')
SAMPLE=np.array([43.21,-44.93,0,22.99,-43.93,-11.35,38.48,-20.49,-38.61,-28.02],dtype=np.float64)

@dataclass(frozen=True)
class QParams:
    bits:int
    method:str
    scale:float
    zero_point:int
    qmin:int
    qmax:int
    offset:float=0.0
    def to_dict(self):return asdict(self)

def parameters(x,bits,method='symmetric'):
    x=np.asarray(x,dtype=np.float64)
    if not isinstance(bits,(int,np.integer)) or isinstance(bits,bool) or not 2<=bits<=16:raise ValueError('bits must be an integer in 2..16')
    if method not in METHODS:raise ValueError('unknown method')
    if not x.size or not np.isfinite(x).all():raise ValueError('nonempty finite values required')
    lo=float(x.min());hi=float(x.max())
    if method=='symmetric':
        m=(1<<(int(bits)-1))-1;scale=max(abs(lo),abs(hi))/m
        return QParams(int(bits),method,scale if scale>0 else 1.0,0,-m,m)
    maxcode=(1<<int(bits))-1
    if method=='endpoint':
        # Literal shifted min-max equation from the assignment; NOT an integer-zero-point grid.
        scale=(hi-lo)/maxcode if hi>lo else 1.0
        return QParams(int(bits),method,scale,int(np.clip(np.rint(-lo/scale),0,maxcode)),0,maxcode,lo)
    lo=min(lo,0.0);hi=max(hi,0.0);scale=(hi-lo)/maxcode if hi>lo else 1.0
    z=int(np.clip(np.rint(-lo/scale),0,maxcode))
    return QParams(int(bits),method,scale,z,0,maxcode)

def quantize(x,p):
    x=np.asarray(x)
    if not np.isfinite(x).all():raise ValueError('nonfinite input')
    value=np.rint((x.astype(np.float64)-p.offset)/p.scale) if p.method=='endpoint' else np.rint(x.astype(np.float64)/p.scale)+p.zero_point
    return np.clip(value,p.qmin,p.qmax).astype(np.int32)

def dequantize(q,p,dtype=np.float32):
    q=np.asarray(q,dtype=np.float64)
    value=q*p.scale+p.offset if p.method=='endpoint' else (q-p.zero_point)*p.scale
    return value.astype(dtype)

def qdq(x,p):return dequantize(quantize(x,p),p)

def pack_codes(q,bits,signed):
    """C-order, first code in least significant bits; little-endian for 16 bits."""
    q=np.asarray(q,dtype=np.int64).ravel()
    low=-(1<<(bits-1)) if signed else 0;high=(1<<(bits-1))-1 if signed else (1<<bits)-1
    if bits not in (2,4,8,16) or np.any(q<low) or np.any(q>high):raise ValueError('unrepresentable code')
    u=q&((1<<bits)-1)
    if bits==16:return u.astype('<u2').tobytes()
    per=8//bits;out=np.zeros((len(q)+per-1)//per,dtype=np.uint8)
    for lane in range(per):
        part=u[lane::per].astype(np.uint8);out[:len(part)]|=part<<(bits*lane)
    return out.tobytes()

def unpack_codes(blob,bits,count,signed):
    if bits not in (2,4,8,16) or count<0 or len(blob)!=(count*bits+7)//8:raise ValueError('invalid packed length')
    if bits==16:u=np.frombuffer(blob,dtype='<u2').astype(np.int32)
    else:
        a=np.frombuffer(blob,dtype=np.uint8);per=8//bits
        u=np.stack([(a>>(i*bits))&((1<<bits)-1) for i in range(per)],1).reshape(-1)[:count].astype(np.int32)
    if signed:u=np.where(u&(1<<(bits-1)),u-(1<<bits),u)
    return u

def integer_dot(xq,wq,zx,zw,bias=0):
    """Exact widened reference with affine zero-point correction."""
    return (np.asarray(xq,dtype=np.int64)-zx)@(np.asarray(wq,dtype=np.int64)-zw).T+np.asarray(bias,dtype=np.int64)

def required_signed_bits(max_abs):
    return max(2,int(max_abs).bit_length()+1)
