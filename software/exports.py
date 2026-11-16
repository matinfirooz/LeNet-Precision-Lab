"""Deterministic, round-trip-checked hardware weight images. NumPy only."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from .quantization import pack_codes,unpack_codes,dequantize,quantize,integer_dot,required_signed_bits,QParams
from .numpy_lenet import LAYERS,INPUT_POINT,prepare_pixels

def export_weights(root,weights,params,codes,activations=None,raw_example=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    wb=bytearray();bb=bytearray();qb=bytearray();rows=[]
    for name in LAYERS:
        key=name+'.weight';q=codes[key];p=params[key];value=weights[key];bias=weights[name+'.bias']
        blob=pack_codes(q,p.bits,p.qmin<0);offset=len(wb);wb.extend(blob)
        restored=unpack_codes(blob,p.bits,q.size,p.qmin<0).reshape(q.shape);assert np.array_equal(restored,q)
        err=float(np.max(abs(dequantize(q,p)-value)));assert err<2e-6,(key,err)
        bo=len(bb);bb.extend(np.asarray(bias,dtype='<f4').tobytes())
        # Binary qparam record is float64 scale, int32 zero, float64 offset: 20 bytes per layer.
        import struct
        qo=len(qb);qb.extend(struct.pack('<did',p.scale,p.zero_point,p.offset))
        row={'layer':name,'shape':list(q.shape),'layout':'OIHW' if q.ndim==4 else 'OI','count':q.size,'weight_byte_offset':offset,'weight_bytes':len(blob),
             'bias_byte_offset':bo,'bias_count':bias.size,'qparam_byte_offset':qo,'qparams':p.to_dict(),'dq_max_error':err}
        if activations is not None:
            ap=activations[INPUT_POINT[name]];ib=np.rint(bias.astype(np.float64)/(ap.scale*p.scale)).astype('<i8')
            (root/(name+'_bias_i64.bin')).write_bytes(ib.tobytes())
            k=int(np.prod(q.shape[1:]));maxx=max(abs(ap.qmin-ap.zero_point),abs(ap.qmax-ap.zero_point));maxw=max(abs(p.qmin-p.zero_point),abs(p.qmax-p.zero_point))
            bound=k*maxx*maxw+int(np.max(abs(ib)))
            row.update(input_qparams=ap.to_dict(),bias_i64_file=name+'_bias_i64.bin',bias_scale=ap.scale*p.scale,accumulator_abs_bound=bound,required_signed_accumulator_bits=required_signed_bits(bound))
        rows.append(row)
    (root/'weights.bin').write_bytes(wb);(root/'biases_fp32.bin').write_bytes(bb);(root/'qparams.bin').write_bytes(qb)
    (root/'weights.hex').write_text(''.join(f'{v:02x}\n' for v in wb))
    np.savez_compressed(root/'codes.npz',**codes)
    metadata={'format_version':1,'author':'Matin Firoozbakht','github':'https://github.com/matinfirooz',
        'packing':'C-order codes, first code in least significant bits; signed two\u2019s complement; little-endian 16-bit; each layer starts on a byte boundary',
        'bias_policy':'FP32 retained for evaluated Q/DQ model; optional rounded INT64 biases are deployment references, not the evaluated bias path',
        'weights_bytes':len(wb),'bias_bytes':len(bb),'qparam_bytes':len(qb),'payload_bytes':len(wb)+len(bb)+len(qb),'fp32_payload_bytes':61706*4,
        'compression_ratio':61706*4/(len(wb)+len(bb)+len(qb)),'layers':rows,'sha256':hashlib.sha256(wb).hexdigest()}
    if activations is not None:
        import struct
        ab=b''.join(struct.pack('<did',v.scale,v.zero_point,v.offset) for v in activations.values())
        (root/'activation_qparams.bin').write_bytes(ab)
        metadata['activations']={k:v.to_dict() for k,v in activations.items()}
        metadata['activation_qparam_bytes']=len(ab)
        metadata['full_qdq_parameter_bytes']=metadata['payload_bytes']+len(ab)
    if raw_example is not None and activations is not None:
        # Central C1 5x5 receptive field for all six output channels.
        ap=activations['input'];wp=params['conv1.weight'];x=prepare_pixels(raw_example[None])[0,0,12:17,12:17].reshape(1,-1)
        xq=quantize(x,ap);wq=codes['conv1.weight'].reshape(6,-1);ib=np.rint(weights['conv1.bias'].astype(np.float64)/(ap.scale*wp.scale)).astype(np.int64)
        acc=integer_dot(xq,wq,ap.zero_point,wp.zero_point,ib)
        expanded=xq.astype(np.int64)@wq.astype(np.int64).T-wp.zero_point*xq.astype(np.int64).sum(1,keepdims=True)-ap.zero_point*wq.astype(np.int64).sum(1)[None]+25*ap.zero_point*wp.zero_point+ib
        assert np.array_equal(acc,expanded)
        np.savez(root/'conv1_integer_golden.npz',input_codes=xq,weight_codes=wq,bias_codes=ib,accumulator=acc,preactivation_reference=acc*(ap.scale*wp.scale))
        metadata['integer_golden']='conv1_integer_golden.npz; padded-input patch origin (12,12), exact INT64 MAC reference, not full integer network verification'
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2));return metadata

def _read_grid(blob,offset,declared):
    """Read the binary grid, rejecting disagreement with the JSON description."""
    p=QParams(**declared)
    if offset<0 or offset+20>len(blob):raise ValueError('truncated grid record')
    scale,zero,shift=struct.unpack_from('<did',blob,offset)
    if (scale,zero,shift)!=(p.scale,p.zero_point,p.offset):raise ValueError('binary grid differs from metadata')
    if not np.isfinite([scale,shift]).all() or scale<=0:raise ValueError('invalid grid scale/offset')
    if p.bits not in (2,4,8,16) or not p.qmin<=zero<=p.qmax:raise ValueError('invalid grid range')
    return QParams(p.bits,p.method,scale,zero,p.qmin,p.qmax,shift)

def load_export(root):
    root=Path(root);m=json.loads((root/'metadata.json').read_text());out={}
    if m.get('format_version')!=1:raise ValueError('unsupported export format')
    w=(root/'weights.bin').read_bytes();b=(root/'biases_fp32.bin').read_bytes();g=(root/'qparams.bin').read_bytes()
    if (len(w),len(b),len(g))!=(m['weights_bytes'],m['bias_bytes'],m['qparam_bytes']):raise ValueError('parameter payload length mismatch')
    if hashlib.sha256(w).hexdigest()!=m['sha256']:raise ValueError('weight digest mismatch')
    if [r['layer'] for r in m['layers']]!=list(LAYERS):raise ValueError('unexpected layer order')
    wo=bo=go=0
    for r in m['layers']:
        if (r['weight_byte_offset'],r['bias_byte_offset'],r['qparam_byte_offset'])!=(wo,bo,go):raise ValueError('non-contiguous parameter offsets')
        p=_read_grid(g,go,r['qparams']);shape=r['shape']
        if not shape or any(not isinstance(v,int) or v<=0 for v in shape) or int(np.prod(shape))!=r['count']:raise ValueError('invalid tensor shape/count')
        size=(r['count']*p.bits+7)//8
        if size!=r['weight_bytes'] or r['bias_count']!=shape[0]:raise ValueError('invalid tensor byte/bias count')
        blob=w[wo:wo+size];codes=unpack_codes(blob,p.bits,r['count'],p.qmin<0).reshape(shape)
        if np.any(codes<p.qmin) or np.any(codes>p.qmax):raise ValueError('code outside declared quantization range')
        tail=r['count']*p.bits%8
        if tail and blob[-1]>>tail:raise ValueError('nonzero packing tail bits')
        out[r['layer']+'.weight']=dequantize(codes,p)
        bias=np.frombuffer(b,dtype='<f4',count=r['bias_count'],offset=bo).copy()
        if not np.isfinite(bias).all():raise ValueError('nonfinite bias')
        out[r['layer']+'.bias']=bias
        wo+=size;bo+=4*r['bias_count'];go+=20
    if (wo,bo,go)!=(len(w),len(b),len(g)):raise ValueError('unaccounted parameter bytes')
    return out,m

def load_activation_params(root,metadata=None):
    from .numpy_lenet import POINTS
    root=Path(root);m=metadata if metadata is not None else json.loads((root/'metadata.json').read_text())
    declared=m.get('activations',{})
    if list(declared)!=list(POINTS):raise ValueError('unexpected activation grid order')
    blob=(root/'activation_qparams.bin').read_bytes()
    if len(blob)!=20*len(POINTS) or len(blob)!=m['activation_qparam_bytes']:raise ValueError('activation grid length mismatch')
    return {name:_read_grid(blob,20*i,declared[name]) for i,name in enumerate(POINTS)}

def export_floating(root,weights,bits):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);dtype='<f2' if bits==16 else '<f4';blob=bytearray();rows=[]
    for name,value in weights.items():
        a=np.asarray(value,dtype=dtype);rows.append({'tensor':name,'shape':list(a.shape),'offset_bytes':len(blob),'bytes':a.nbytes});blob.extend(a.tobytes())
    (root/'parameters.bin').write_bytes(blob)
    m={'format':'IEEE floating point, little-endian, C-order','bits':bits,'payload_bytes':len(blob),'tensors':rows,'sha256':hashlib.sha256(blob).hexdigest()}
    (root/'metadata.json').write_text(json.dumps(m,indent=2));return m

def export_native(root,model):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);rows=[];size=0
    for name in LAYERS:
        layer=getattr(model.net,name);w=layer.weight();code=w.int_repr().numpy().astype('i1');scale=w.q_per_channel_scales().numpy().astype('<f8');zero=w.q_per_channel_zero_points().numpy().astype('<i4');bias=layer.bias().detach().numpy().astype('<f4')
        for part,a in [('weight',code),('scale',scale),('zero',zero),('bias',bias)]:
            (root/f'{name}_{part}.bin').write_bytes(a.tobytes());size+=a.nbytes
        rows.append({'layer':name,'shape':list(code.shape),'weight_dtype':'int8','weight_qscheme':'per_output_channel_affine','scale_dtype':'float64 little-endian','zero_dtype':'int32 little-endian','bias_dtype':'float32 little-endian','output_scale':float(layer.scale),'output_zero_point':int(layer.zero_point),'axis':0})
    import struct
    activation=[(float(model.quant.scale.item()),int(model.quant.zero_point.item()))]+[(r['output_scale'],r['output_zero_point']) for r in rows]
    ab=b''.join(struct.pack('<di',s,z) for s,z in activation);(root/'activation_qparams.bin').write_bytes(ab);size+=len(ab)
    m={'format':'Converted PyTorch CPU INT8 layer payloads; tanh and avgpool quantization follow the pinned PyTorch backend','input_scale':float(model.quant.scale.item()),'input_zero_point':int(model.quant.zero_point.item()),'payload_bytes':size,'layers':rows,'note':'This is a separate per-channel backend result, not the custom per-tensor PTQ grid.'}
    (root/'metadata.json').write_text(json.dumps(m,indent=2));return m
