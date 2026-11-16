"""PyTorch PTQ Q/DQ experiments plus a real converted CPU INT8 model."""
import copy
import torch
from torch import nn
from torch.ao.quantization import (FakeQuantize,MinMaxObserver,QuantStub,DeQuantStub,
                                  QConfig,PerChannelMinMaxObserver,prepare,convert)
from .model import LeNet
from .numpy_lenet import POINTS,LAYERS
from .quantization import QParams

def make_fake(bits,method):
    maximum=2**(bits-1)-1 if method=='symmetric' else 2**bits-1
    dtype=torch.qint32 if bits==16 else (torch.qint8 if method=='symmetric' else torch.quint8)
    return FakeQuantize(observer=MinMaxObserver,dtype=dtype,
        qscheme=torch.per_tensor_symmetric if method=='symmetric' else torch.per_tensor_affine,
        quant_min=-maximum if method=='symmetric' else 0,quant_max=maximum)

def describe(f,bits,method):
    return QParams(bits,method,float(f.scale.item()),int(f.zero_point.item()),f.quant_min,f.quant_max)

def weight_qdq(model,bits,method):
    model=copy.deepcopy(model).eval();params={};codes={}
    with torch.no_grad():
        for name,module in model.named_modules():
            if isinstance(module,(nn.Conv2d,nn.Linear)):
                f=make_fake(bits,method);value=f(module.weight.detach());f.disable_observer()
                p=describe(f,bits,method);q=torch.clamp(torch.round(value/f.scale)+f.zero_point,p.qmin,p.qmax).to(torch.int32)
                module.weight.copy_(value);params[name+'.weight']=p;codes[name+'.weight']=q.numpy()
    return model,params,codes

def calibrated_activations(ranges,bits,method):
    params={}
    for name in POINTS:
        f=make_fake(bits,method);f(torch.tensor(ranges[name],dtype=torch.float32));f.disable_observer()
        params[name]=describe(f,bits,method)
    return params

class QDQLeNet(nn.Module):
    """Integer grids are simulated; convolution, pooling and tanh still compute in FP32."""
    def __init__(self,net,activations=None):super().__init__();self.net=net;self.activations=activations
    def at(self,name,x):
        if self.activations is None:return x
        p=self.activations[name]
        return torch.fake_quantize_per_tensor_affine(x,p.scale,p.zero_point,p.qmin,p.qmax)
    def forward(self,x):
        m=self.net;x=self.at('input',x);x=self.at('t1',torch.tanh(m.conv1(x)))
        x=self.at('p1',torch.tanh(nn.functional.avg_pool2d(x,2)))
        x=self.at('t3',torch.tanh(m.conv3(x)));x=self.at('p3',torch.tanh(nn.functional.avg_pool2d(x,2)))
        x=self.at('t5',torch.tanh(m.conv5(x)).flatten(1));x=self.at('t6',torch.tanh(m.fc6(x)))
        return self.at('logits',m.fc7(x))

class NativeINT8(nn.Module):
    def __init__(self,net=None):
        super().__init__();self.quant=QuantStub();self.net=LeNet() if net is None else copy.deepcopy(net);self.dequant=DeQuantStub()
    def forward(self,x):return self.dequant(self.net(self.quant(x)))

def convert_native(net,calibration,engine='x86'):
    if engine not in torch.backends.quantized.supported_engines:raise RuntimeError('INT8 engine unavailable: '+engine)
    torch.backends.quantized.engine=engine;m=NativeINT8(net).eval()
    m.qconfig=QConfig(activation=MinMaxObserver.with_args(dtype=torch.quint8,qscheme=torch.per_tensor_affine,quant_min=0,quant_max=255),
                     weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_channel_symmetric,quant_min=-128,quant_max=127))
    prepare(m,inplace=True)
    with torch.inference_mode():
        for x in calibration:m(x)
    convert(m,inplace=True);return m

def load_qdq(path):
    d=torch.load(path,map_location='cpu',weights_only=True);m=LeNet();m.load_state_dict(d['state_dict'])
    ap={k:QParams(**v) for k,v in d['activations'].items()} if d.get('activations') else None
    return QDQLeNet(m,ap).eval()
