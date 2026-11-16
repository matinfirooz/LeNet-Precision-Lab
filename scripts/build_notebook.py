"""Build a complete, readable notebook with written answers, live checks and full source."""
import json,re,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from software.reporting import test_count
N_TESTS=test_count(ROOT)
cells=[]
def md(s):cells.append({'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},'source':s.replace('@TEST_COUNT@',str(N_TESTS)).splitlines(True)})
def code(s):cells.append({'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},'source':s.replace('@TEST_COUNT@',str(N_TESTS)).splitlines(True),'outputs':[],'execution_count':None})
md('''# LeNet Quantization, Pruning and Hardware Weight Export

**Matin Firoozbakht**  
GitHub: [@matinfirooz](https://github.com/matinfirooz)

Complete answers to the supplied HW4 specification, with genuine executed outputs, the complete source, all precision results and hardware-style figures. Every reported accuracy uses all **10,000 official Fashion-MNIST test images**.

For execution, keep this notebook inside the extracted `HW4_Matin_Firoozbakht` project. The embedded outputs and source listings can be read directly. The supplied data and models allow offline evaluation. Default execution validates evidence, recomputes FP32 evaluation, demonstrates the manual methods, decodes packed weights and runs all @TEST_COUNT@ tests.
''')
code('''from pathlib import Path
import os, sys, json, re, subprocess, importlib.metadata
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT = Path.cwd()
assert (ROOT/'software/quantization.py').exists(), 'Run inside the extracted project directory.'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
RETRAIN = False
RECOMPUTE_EXPERIMENTS = False
RUN_TESTS = True

if RETRAIN: RECOMPUTE_EXPERIMENTS = True
if RECOMPUTE_EXPERIMENTS:
    steps = (['train.py'] if RETRAIN else []) + ['run_manual.py','run_pytorch.py','run_pruning.py','benchmark.py','qualify_packed.py','render_figures.py']
    for step in steps:
        subprocess.run([sys.executable, str(ROOT/'scripts'/step)], cwd=ROOT, check=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'], cwd=ROOT, check=True)
    subprocess.run([sys.executable,'scripts/verify_evidence.py','--record'], cwd=ROOT, check=True)
    subprocess.run([sys.executable,'scripts/write_documentation.py'], cwd=ROOT, check=True)

import numpy as np
import pandas as pd
import torch
from IPython.display import display, Markdown, Image, Code
from software.evidence import validate
report = validate(ROOT)
assert report['status']=='passed', report
print('Author: Matin Firoozbakht | GitHub: @matinfirooz')
print('Saved experiment evidence:', report)
print('Python',sys.version.split()[0], '| NumPy',np.__version__, '| PyTorch',torch.__version__)
torch.set_num_threads(4)
''')
# Dynamic reading makes narrative update when experiments are recomputed.
code('''def answer_section(number):
    text=(ROOT/'COMPLETE_ANSWERS.md').read_text()
    match=re.search(rf'(?m)^## {number}\\.',text)
    assert match, number
    next_section=re.search(r'(?m)^## \\d+\\.',text[match.end():])
    end=match.end()+next_section.start() if next_section else len(text)
    display(Markdown(text[match.start():end]))
def figure(relative):
    display(Image(filename=str(ROOT/relative)))
'''.replace('\\\\','\\'))
for n in [1,2,3]:
    code(f'answer_section({n})')
    if n==3:
        code('''from software.quantization import SAMPLE, parameters, quantize, dequantize
sample_rows=[]
for method in ['symmetric','affine','endpoint']:
    p=parameters(SAMPLE,8,method)
    q=quantize(SAMPLE,p)
    print(method, '| scale =',p.scale,'| zero =',p.zero_point,'| codes =',q.tolist())
    sample_rows.append(pd.DataFrame({'original':SAMPLE,'code':q,'reconstructed':dequantize(q,p,np.float64),'method':method}))
display(pd.concat(sample_rows,ignore_index=True))
assert quantize(SAMPLE,parameters(SAMPLE,8,'affine'))[3]==197
figure('results/figures/worked_example.png')
''')
code('answer_section(4)')
code('''from software.data import load_idx, split_indices, CLASSES
from software.model import LeNet, prepare_pixels
from scripts.train import evaluate
raw, labels=load_idx(ROOT/'data/fashion','t10k')
checkpoint=torch.load(ROOT/'software/lenet_fashion_mnist.pt',map_location='cpu',weights_only=True)
model=LeNet().eval(); model.load_state_dict(checkpoint['state_dict'])
measured, logits=evaluate(model,prepare_pixels(raw),torch.tensor(labels,dtype=torch.long))
expected=json.loads((ROOT/'results/software/training_summary.json').read_text())['test']
assert measured['correct']==expected['correct']
print('Recomputed full-test FP32:', measured)
print('Parameters:',sum(p.numel() for p in model.parameters()),'| Selected epoch:',checkpoint['epoch'])
predictions=logits.argmax(1)
confusion=np.zeros((10,10),dtype=np.int32)
np.add.at(confusion,(labels,predictions),1)
display(pd.DataFrame(confusion,index=CLASSES,columns=CLASSES).rename_axis('True class; columns are predictions'))
display(pd.DataFrame({'class':CLASSES,'images':np.bincount(labels,minlength=10),'correct':np.diag(confusion),'accuracy_percent':np.diag(confusion)/np.bincount(labels,minlength=10)*100}))
figure('hardware/figures/lenet_architecture.png')
figure('results/figures/training.png')
''')
code('answer_section(5)')
code('''manual=json.loads((ROOT/'results/quantization/manual_results.json').read_text())
assert manual['neural_network_imports']==[]
assert manual['baseline']['torch_prediction_disagreements']==0
print('Independent NumPy process:',manual['baseline'])
print('Neural-network package imports:',manual['neural_network_imports'])
probe=subprocess.run([sys.executable,'-c',"import sys; from software.numpy_lenet import forward; assert not any(n in sys.modules for n in ['torch','tensorflow','jax','keras']); print('Fresh NumPy-only import probe passed.')"],capture_output=True,text=True,cwd=ROOT,check=True)
print(probe.stdout.strip())
display(pd.DataFrame(manual['results']))
figure('results/figures/manual_accuracy.png')
''')
code('answer_section(6)')
code('''framework=json.loads((ROOT/'results/quantization/pytorch_results.json').read_text())
display(pd.DataFrame(framework['results']))
display(pd.DataFrame([framework['native_int8']]))
display(pd.DataFrame(framework['native_operators']))
figure('results/figures/pytorch_accuracy.png')
''')
code('answer_section(7)')
code("figure('results/figures/compression_accuracy.png')")
code('answer_section(8)')
code('''pruning=json.loads((ROOT/'results/pruning/pruning_results.json').read_text())
display(pd.DataFrame(pruning['rates']))
print('Recovered 75% sparse:',pruning['recovered']['correct'],'/ 10000')
print('Compacted:',pruning['compact']['parameters'],'parameters,',pruning['compact']['dense_macs'],'dense MACs/image')
print('Compaction masked-reference maximum error:',pruning['compact']['masked_equivalence_max_error'])
figure('results/figures/pruning.png')
figure('hardware/figures/pruning_compaction.png')
''')
code('answer_section(9)')
code('''from software.exports import load_export
from software.quantization import integer_dot
export=ROOT/'hardware/exports/affine_int8'
restored, metadata=load_export(export)
state=torch.load(ROOT/'models/affine_int8_weights_only.pt',map_location='cpu',weights_only=True)['state_dict']
for name, value in restored.items():
    np.testing.assert_array_equal(value,state[name].numpy())
print('All five layers and biases reconstructed exactly from packed bytes.')
display(pd.DataFrame([{k:v for k,v in layer.items() if k in ['layer','shape','count','weight_byte_offset','weight_bytes','required_signed_accumulator_bits']} for layer in metadata['layers']]))
golden=np.load(export/'conv1_integer_golden.npz')
ap=metadata['activations']['input']; wp=metadata['layers'][0]['qparams']
actual=integer_dot(golden['input_codes'],golden['weight_codes'],ap['zero_point'],wp['zero_point'],golden['bias_codes'])
np.testing.assert_array_equal(actual,golden['accumulator'])
assert np.unique(golden['input_codes']).size>1
print('Central 5x5 input codes:',golden['input_codes'].reshape(5,5))
print('Exact centered INT64 MAC outputs:',actual.tolist())
print('Scaled preactivations:',golden['preactivation_reference'].tolist())
figure('hardware/figures/packed_weights.png')
figure('hardware/figures/quantized_datapath.png')
''')
code("""from software.packed_runtime import PackedLeNet
packed=PackedLeNet(ROOT/'hardware/exports/affine_int8')
probabilities=packed.probabilities(raw[:4])
display(pd.DataFrame(probabilities,columns=CLASSES))
packed_report=json.loads((ROOT/'results/quantization/packed_results.json').read_text())
assert packed_report['neural_network_imports']==[]
display(pd.DataFrame([{k:v for k,v in row.items() if k!='input_sha256'} for row in packed_report['results']]))
print('Full-test packed configurations evaluated:',len(packed_report['results']))
print('Prediction differences from saved PyTorch references:',sum(row['saved_torch_prediction_disagreements'] for row in packed_report['results']))
""")
code('answer_section(10)')
code("figure('results/figures/latency.png')")
code('answer_section(11)')
code('''if RUN_TESTS:
    test_run=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,text=True,capture_output=True)
    print(test_run.stdout+test_run.stderr)
    assert test_run.returncode==0, 'A verification test failed.'
print('Evidence after notebook evaluation:',validate(ROOT))
''')
code('answer_section(12)')
md('''# Complete implementation source

The listings below are the exact sources used by this notebook and included in the ZIP. Step 1 is isolated in the manual quantization, NumPy inference and data modules. Training, PyTorch PTQ and pruning are separate. These listings are displayed as source rather than silently rerunning long training jobs.
''')
files=['software/quantization.py','software/numpy_lenet.py','software/data.py','software/model.py','scripts/train.py','scripts/run_manual.py','software/torch_quantization.py','scripts/run_pytorch.py','software/exports.py','software/packed_runtime.py','scripts/infer_packed.py','scripts/qualify_packed.py','software/pruning.py','scripts/run_pruning.py','scripts/benchmark.py','software/evidence.py','tests/test_quantization.py','tests/test_project.py','software/figures.py','software/schematic_symbols.py','software/reporting.py','scripts/run_all.py','scripts/execute_notebook.py']
for f in files:
    md('## Source: '+f)
    code(f"display(Code(filename=str(ROOT/{f!r}), language='python'))")
md('''# Reproduction and submission files

The ZIP contains this notebook, the main README, complete written answers, all source modules, models, packed hardware weights, official compressed data, tests and PNG/SVG figures. Use the ordered commands in [README.md](README.md) or `python scripts/run_all.py --train` to regenerate all deliverables. The notebook's first cell exposes optional full-experiment and retraining switches.

**Author: Matin Firoozbakht · GitHub: [@matinfirooz](https://github.com/matinfirooz)**
''')
notebook={'cells':cells,'metadata':{'authors':[{'name':'Matin Firoozbakht','github':'https://github.com/matinfirooz'}],'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'title':'LeNet Quantization and Pruning: Complete Answers'},'nbformat':4,'nbformat_minor':5}
(ROOT/'HW4_Matin_Firoozbakht.ipynb').write_text(json.dumps(notebook,indent=1,ensure_ascii=False))
print('Built',len(cells),'cells.')
