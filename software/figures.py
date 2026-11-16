"""Publication-style plots and exact vector diagrams generated from saved measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle,FancyArrowPatch
from .schematic_symbols import Circuit,BLACK,GRAY,BLUE,CONTROL
from .quantization import SAMPLE,parameters,quantize,dequantize

C={'symmetric':'#226682','affine':'#C26629','endpoint':'#7A5292'}

def save(fig,root,name):
    fig.savefig(root/f'results/figures/{name}.png',dpi=160,bbox_inches='tight',facecolor='white')
    fig.savefig(root/f'results/figures/{name}.svg',bbox_inches='tight',facecolor='white');plt.close(fig)

def hardware(root):
    c=Circuit('LeNet-5: precision boundaries and layer organization','Dense C3, tanh activations, average pooling; all shapes exclude the batch dimension',w=16,h=8)
    nodes=[]
    specs=[('Input','1 × 32 × 32',1,5.4,2.2),('C1 + tanh','6 × 28 × 28',4.3,5.4,2.4),('S2 + tanh','6 × 14 × 14',8,5.4,2.4),('C3 + tanh','16 × 10 × 10',11.7,5.4,2.7),
      ('S4 + tanh','16 × 5 × 5',11.7,2.7,2.7),('C5 + tanh','120 × 1 × 1',8,2.7,2.4),('F6 + tanh','84',4.3,2.7,2.4),('F7 logits','10',1,2.7,2.2)]
    for name,shape,x,y,w in specs:nodes.append(c.block(x,y,w,1.15,name,shape,fill='#F3F7F8'))
    for i in [0,1,2]:c.wire([nodes[i]['R'],nodes[i+1]['L']],label='Q/DQ' if i>0 else 'pixels / 255',bits=None if i else 8)
    c.wire([nodes[3]['B'],nodes[4]['T']],label='Q/DQ')
    for i in [4,5,6]:c.wire([nodes[i]['L'],nodes[i+1]['R']],label='Q/DQ')
    c.text(8,1.95,'61,706 parameters  |  416,520 dense MACs / image  |  10 raw logits → argmax',11,bold=True)
    c.text(8,1.5,'Weight-only: WbA32  •  Activation study: WbAb at input, tanh/pool outputs and logits',9.5)
    c.text(8,1.08,'Arrows show tensor flow; bus widths in this view are illustrative precision choices, not an implemented RTL interface.',8.5,color=GRAY)
    c.footer('Authored model diagram');c.save(root,'hardware/figures/lenet_architecture.png')

    c=Circuit('Quantized MAC: explicit affine correction and widened accumulation','Mapping reference for packed integer weights; the evaluated Q/DQ model retains FP32 arithmetic and nonlinear functions',w=16,h=8)
    xr=c.reg(.8,4.2,1.65,1.1,'x code','b',rst=None)
    wr=c.reg(.8,2.3,1.65,1.1,'w code','b',rst=None)
    sx=c.alu(4,4.75,'−',.32);sw=c.alu(4,2.85,'−',.32)
    c.wire([xr['Q'],sx['L']],label='qx',bits=16);c.wire([wr['Q'],sw['L']],label='qw',bits=16)
    c.wire([(4,6),sx['T']],label='zx',control=True);c.wire([(4,1.65),sw['B']],label='zw',control=True)
    mul=c.alu(6.25,3.8,'×',.37)
    c.wire([sx['R'],(5.3,4.75),(5.3,3.8),mul['L']],label='qx − zx')
    c.wire([sw['R'],(6.25,2.85),mul['B']],label='qw − zw')
    add=c.alu(8.0,3.8,'+',.3);mux=c.mux(9.1,3.8,.65,1.15,sel='init');acc=c.reg(10.2,3.25,1.7,1.1,'ACC',64,ce='step',rst='clear')
    c.wire([mul['R'],add['L']],bits=34,label='product')
    c.wire([add['R'],(8.65,3.8),(8.65,mux['0'][1]),mux['0']],bits=64)
    c.wire([mux['Q'],acc['D']],bits=64)
    c.wire([acc['Q'],(12.1,3.8),(12.1,5.3),(8.0,5.3),add['T']],bits=64,label='feedback')
    out=c.block(12.8,3.15,2.6,1.25,'Scale + tanh','FP32 in this study',fill='#F6F6F2')
    c.dot(12.1,3.8);c.wire([(12.1,3.8),out['L']],bits=64)
    bias=c.block(5.8,1.25,3.1,.95,'Bias code','round(bias / (sx · sw))',size=9)
    c.wire([bias['R'],(9.98,1.725),(9.98,4.5),(8.8,4.5),(8.8,mux['1'][1]),mux['1']],bits=64)
    c.text(8,6.65,'Accumulator bound = K · max|qx − zx| · max|qw − zw| + |bias code|',11,bold=True)
    c.text(8,1.03,'Per-layer bounds and exact INT64 C1 golden vectors are exported; these registers depict a possible clocked mapping.',8.5,color=GRAY)
    c.footer('Authored hardware mapping reference');c.save(root,'hardware/figures/quantized_datapath.png')

    c=Circuit('Weight image: packed codes, typed metadata and decoder','Byte-addressed storage, C-order OIHW/OI tensors; each layer begins at a byte boundary',w=16,h=8)
    mem=c.mem(.8,2.8,3.7,3.4,'weights.bin',['byte 0: [q1 | q0]','byte 1: [q3 | q2]','INT4: q0 in bits [3:0]'],8,30735,clocked=False)
    # This diagram illustrates INT4 weights; exact weight bytes exclude bias and qparams.
    c.text(2.65,6.5,'Illustration: INT4 packing',10,bold=True)
    decode=c.block(6.2,4,3,1.4,'Unpack code','shift, mask, sign extend',fill='#F3F7F8')
    c.wire([mem['Q'],(5.35,mem['Q'][1]),(5.35,4.7),decode['L']],bits=8,label='byte')
    c.wire([decode['R'],(10.2,4.7)],bits=4,label='q')
    dq=c.block(10.2,4,4.4,1.4,'Dequantize','w = (q − z) · scale',fill='#F3F7F8')
    meta=c.block(5.9,1.6,4.2,1.25,'qparams.bin','float64 scale | int32 z | float64 offset',size=9)
    bias=c.block(11.1,1.6,3.5,1.25,'biases_fp32.bin','236 IEEE FP32 values',size=9)
    c.wire([meta['T'],(8,3.35),(12.4,3.35),dq['B']],control=True,label='grid parameters')
    c.text(8,6.9,'INT2: 4 codes / byte     INT4: 2 codes / byte     INT8: 1 code / byte     INT16: 2 bytes / code',10,bold=True)
    c.text(8,1.03,'Metadata records layer shape, offsets, signedness, accumulator bounds and SHA-256. Decode round trips are checked.',8.8,color=GRAY)
    c.footer('Authored binary layout diagram');c.save(root,'hardware/figures/packed_weights.png')

    c=Circuit('Pruning: masks and physical channel compaction','Fixed global L1 rates plus a separate L1 channel selection experiment; three recovery epochs use training and validation only',w=16,h=8)
    dense=c.block(.8,3.3,3.5,1.5,'Dense LeNet','6 / 16 / 120 / 84 channels',fill='#F3F7F8')
    mask=c.block(6,4.9,4.3,1.35,'Magnitude masks','25 / 50 / 75 / 90% weights',fill='#F6F6F2')
    sparse=c.block(11.9,4.9,3.3,1.35,'CSR export','FP32 values + indices')
    compact=c.block(6,2,4.3,1.35,'Keep + propagate','4 / 12 / 90 / 63 channels',fill='#F3F7F8')
    small=c.block(11.9,2,3.3,1.35,'Compact LeNet','34,779 parameters')
    c.wire([dense['R'],(5.05,4.05),(5.05,5.575),mask['L']])
    c.wire([(5.05,4.05),(5.05,2.675),compact['L']]);c.dot(5.05,4.05)
    c.wire([mask['R'],sparse['L']],label='zero weights');c.wire([compact['R'],small['L']],label='slice tensors')
    c.text(8,1.25,'Sparse masks keep dense dimensions and 416,520 MACs; compaction reduces dense MACs to 231,700 / image.',10,bold=True)
    c.footer('Authored pruning diagram');c.save(root,'hardware/figures/pruning_compaction.png')


def main(root):
    root=Path(root);(root/'results/figures').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelsize':10,'font.size':10,'svg.fonttype':'none'})
    manual=json.loads((root/'results/quantization/manual_results.json').read_text());pt=json.loads((root/'results/quantization/pytorch_results.json').read_text());pr=json.loads((root/'results/pruning/pruning_results.json').read_text());history=json.loads((root/'results/software/training_history.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for key,label in [('train_accuracy','Training'),('validation_accuracy','Validation')]:ax[0].plot([r['epoch'] for r in history],[r[key]*100 for r in history],label=label,lw=2)
    for key,label in [('train_cross_entropy','Training'),('validation_cross_entropy','Validation')]:ax[1].plot([r['epoch'] for r in history],[r[key] for r in history],label=label,lw=2)
    for a in ax:a.set_xlabel('Epoch');a.grid(alpha=.18);a.legend(frameon=False)
    ax[0].set_ylabel('Accuracy (%)');ax[1].set_ylabel('Cross-entropy');fig.suptitle('FP32 training: checkpoint selected by validation cross-entropy',fontweight='bold');fig.tight_layout();save(fig,root,'training')
    for source,name,title,bits in [(manual,'manual_accuracy','Manual NumPy quantization',[4,8,16]),(pt,'pytorch_accuracy','PyTorch post-training quantization',[2,4,8,16])]:
        fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
        for ax,mode,label in zip(axes,['weights_only','weights_activations'],['Weight-only WbA32','Weights + activations WbAb']):
            for method in (['symmetric','affine','endpoint'] if source is manual else ['symmetric','affine']):
                rows=sorted([r for r in source['results'] if r['method']==method and r['mode']==mode],key=lambda r:r['bits'])
                ax.plot([r['bits'] for r in rows],[r['accuracy']*100 for r in rows],'-o',label=method.capitalize(),c=C[method],lw=2)
            ax.axhline(manual['baseline']['accuracy']*100,c='#58606B',ls='--',lw=1,label=f"FP32 {manual['baseline']['accuracy']*100:.2f}%");ax.set_xticks(bits);ax.set_xlabel('Integer bit-width');ax.set_title(label,fontsize=11);ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)
        axes[0].set_ylabel('Test accuracy (%)');fig.suptitle(title+' · 10,000 test images',fontweight='bold');fig.tight_layout();save(fig,root,name)
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    for method in ['symmetric','affine']:
        rows=[r for r in pt['results'] if r['method']==method and r['mode']=='weights_only']
        ax.plot([r['payload_bytes']/1024 for r in rows],[r['accuracy']*100 for r in rows],'-o',c=C[method],label=method.capitalize())
        for r in rows:ax.annotate(f"INT{r['bits']}",(r['payload_bytes']/1024,r['accuracy']*100),xytext=(6,-13 if method=='affine' else 7),textcoords='offset points',fontsize=8)
    for name in ['FP32','FP16']:
        r=next(r for r in pt['results'] if r['name']==name);ax.scatter(r['payload_bytes']/1024,r['accuracy']*100,c='#36424D',s=40);ax.annotate(name,(r['payload_bytes']/1024,r['accuracy']*100),xytext=(-30,-20),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Parameter payload (KiB), including biases and binary grid records',ylabel='Test accuracy (%)',title='Accuracy and actual packed parameter storage');ax.grid(alpha=.18);ax.legend(frameon=False);fig.tight_layout();save(fig,root,'compression_accuracy')
    fig,ax=plt.subplots(1,2,figsize=(11,4.3))
    rows=pr['rates'];ax[0].plot([r['weight_sparsity']*100 for r in rows],[r['accuracy']*100 for r in rows],'-o',color=BLUE,label='No recovery')
    ax[0].scatter([75],[pr['recovered']['accuracy']*100],color=CONTROL,label='3-epoch recovery',s=70,zorder=5)
    ax[0].set(xlabel='Weight sparsity (%)',ylabel='Test accuracy (%)',title='Global L1 magnitude pruning');ax[0].legend(frameon=False)
    ax[1].plot([r['weight_sparsity']*100 for r in rows],[r['sparse_payload_bytes']/1024 for r in rows],'-o',color=BLUE,label='CSR payload')
    ax[1].axhline(246824/1024,color=CONTROL,ls='--',label='Dense FP32 payload');ax[1].set(xlabel='Weight sparsity (%)',ylabel='Parameter payload (KiB)',title='Sparse index overhead is counted');ax[1].legend(frameon=False)
    for a in ax:a.grid(alpha=.18)
    fig.tight_layout();save(fig,root,'pruning')
    fig,ax=plt.subplots(figsize=(9,4.1));ind=np.arange(len(SAMPLE));ax.plot(ind,SAMPLE,'o-',label='Original',c=BLACK)
    for method in ['symmetric','affine','endpoint']:
        p=parameters(SAMPLE,4,method);ax.plot(ind,dequantize(quantize(SAMPLE,p),p),'x--',label=method.capitalize(),c=C[method])
    ax.set(xlabel='Sample element',ylabel='Value',title='Worked sample: 4-bit reconstruction');ax.grid(alpha=.18);ax.legend(frameon=False,ncol=4);fig.tight_layout();save(fig,root,'worked_example')
    latency_path=root/'results/software/latency.json'
    if latency_path.exists():
        data=json.loads(latency_path.read_text());fig,ax=plt.subplots(figsize=(9,4.5));rows=data['results'];labels=['FP32','FP16','Native INT8','75% sparse\ndense kernels','Compacted\nFP32'];bars=ax.bar(labels,[r['median_us'] for r in rows],color=[BLUE,'#6296AA',CONTROL,'#87949B','#548468'])
        for bar,r in zip(bars,rows):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{r['median_us']:.1f}",ha='center',va='bottom',fontsize=9)
        ax.set(ylabel='Median latency (µs / image)',title='Measured CPU inference · batch 1 · one thread');ax.set_ylim(0,max(r['median_us'] for r in rows)*1.18);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);fig.tight_layout();save(fig,root,'latency')
    hardware(root)

if __name__=='__main__':main(Path(__file__).resolve().parents[1])
