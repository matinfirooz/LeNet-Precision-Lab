"""Generate current-design documentation directly from completed experiment records."""
import ast,csv,json
from pathlib import Path

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(r):return f"{100*r['accuracy']:.2f}"
def load(r,p):return json.loads((r/p).read_text())

def test_count(root):
    return sum(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')
               for p in (Path(root)/'tests').glob('test_*.py') for n in ast.walk(ast.parse(p.read_text())))

def generate(root):
    r=Path(root);tr=load(r,'results/software/training_summary.json');ma=load(r,'results/quantization/manual_results.json');pt=load(r,'results/quantization/pytorch_results.json');pr=load(r,'results/pruning/pruning_results.json');la=load(r,'results/software/latency.json')
    n_tests=test_count(r)
    packed=load(r,'results/quantization/packed_results.json')
    def q(bits,method='symmetric',mode='weights_only'):
        return next(v for v in pt['results'] if v['bits']==bits and v['method']==method and v['mode']==mode)
    def mq(bits,method,mode):
        return next(v for v in ma['results'] if v['bits']==bits and v['method']==method and v['mode']==mode)
    fp32=next(v for v in pt['results'] if v['name']=='FP32')
    fp16=next(v for v in pt['results'] if v['name']=='FP16')
    compact=pr['compact'];weight_sparsity={v['rate']:v for v in pr['rates']}
    analysis=f"""**How does quantization affect accuracy?** FP32 scores {pct(fp32)}% and FP16 {pct(fp16)}%. At sixteen bits, symmetric and affine weight-only accuracy are {pct(q(16))}% and {pct(q(16,'affine'))}%. At eight bits they are {pct(q(8))}% and {pct(q(8,'affine'))}%, with W8A8 at {pct(q(8,mode='weights_activations'))}% and {pct(q(8,'affine','weights_activations'))}%. At two bits, the weight-only results are {pct(q(2))}% symmetric and {pct(q(2,'affine'))}% affine. Small differences on a single test set do not establish a repeatable accuracy improvement.

**What is the compression/accuracy trade-off?** FP32 parameter storage is {fp32['payload_bytes']:,} bytes. Packed weight-only payloads at INT16/8/4/2 are {', '.join(format(q(b)['payload_bytes'],',') for b in [16,8,4,2])} bytes. Their compression ratios are {', '.join(format(q(b)['compression_ratio'],'.2f')+'×' for b in [16,8,4,2])}, including biases and weight grid records. Four-bit symmetric weight-only inference scores {pct(q(4))}% with {q(4)['compression_ratio']:.2f}× compression. WbAb additionally stores activation grid records; its complete byte counts are included in the result files. JSON, serialization overhead and activation working buffers are separate quantities.

**Which method works better?** Compare each policy separately. At four bits, symmetric weight-only accuracy is {pct(q(4))}% and affine is {pct(q(4,'affine'))}%. W4A4 scores {pct(q(4,mode='weights_activations'))}% symmetric and {pct(q(4,'affine','weights_activations'))}% affine. The literal endpoint grid scores {pct(mq(4,'endpoint','weights_only'))}% weight-only and {pct(mq(4,'endpoint','weights_activations'))}% W4A4. Outliers, per-tensor range selection and activation rounding affect these outcomes. The native per-channel configuration is reported independently.

**At what bit-width does accuracy significantly degrade?** Use a descriptive threshold of a one-percentage-point loss from FP32. At four bits, symmetric and affine weight-only losses are {(fp32['accuracy']-q(4)['accuracy'])*100:.2f} and {(fp32['accuracy']-q(4,'affine')['accuracy'])*100:.2f} points. The corresponding W4A4 losses are {(fp32['accuracy']-q(4,mode='weights_activations')['accuracy'])*100:.2f} and {(fp32['accuracy']-q(4,'affine','weights_activations')['accuracy'])*100:.2f} points. The first width crossing this threshold when stepping through 16, 8, 4 and 2 bits is: """
    crossings=[]
    for method in ['symmetric','affine']:
        for mode in ['weights_only','weights_activations']:
            first=next((str(b) for b in [16,8,4,2] if (fp32['accuracy']-q(b,method,mode)['accuracy'])*100>=1),'none')
            crossings.append(f'{method} {mode}: {first}')
    analysis+='; '.join(crossings)+'. This is an explanatory convention, not a hypothesis test or a bit-width selected to optimize test accuracy.\n\n'
    analysis+='**Can reduced memory be claimed as a speedup?** Q/DQ uses FP32 operators. Native low-precision kernels and sparse formats also have dispatch, packing and indexing costs. The separate CPU measurements describe the tested host and shapes; parameter compression alone is not a latency measurement.\n'
    packed_table=table(['Bits','Method','Policy','Accuracy (%)','Prediction differences vs PyTorch'],[[v['bits'],v['method'],v['mode'],pct(v),v['saved_torch_prediction_disagreements']] for v in packed['results']])
    manual=table(['Bits','Method','WbA32 correct / 10,000','WbA32 accuracy (%)','WbAb accuracy (%)'],[[bits,m,str(next(v for v in ma['results'] if v['bits']==bits and v['method']==m and v['mode']=='weights_only')['correct']),pct(next(v for v in ma['results'] if v['bits']==bits and v['method']==m and v['mode']=='weights_only')),pct(next(v for v in ma['results'] if v['bits']==bits and v['method']==m and v['mode']=='weights_activations'))] for bits in [16,8,4] for m in ['symmetric','affine','endpoint']])
    rows=[]
    for name in ['FP32','FP16']:
        a=next(v for v in pt['results'] if v['name']==name);rows.append([name,'floating point',a['correct'],pct(a),'—',a['payload_bytes'],f"{a['compression_ratio']:.2f}×"])
    for bits in [16,8,4,2]:
        for m in ['symmetric','affine']:
            a=next(v for v in pt['results'] if v['bits']==bits and v['method']==m and v['mode']=='weights_only');b=next(v for v in pt['results'] if v['bits']==bits and v['method']==m and v['mode']=='weights_activations')
            rows.append([f'INT{bits}',m,a['correct'],pct(a),pct(b),a['payload_bytes'],f"{a['compression_ratio']:.2f}×"])
    precision=table(['Precision','Method','WbA32 correct','WbA32 accuracy (%)','WbAb accuracy (%)','WbA32 payload (bytes)','Compression'],rows)
    pruning=table(['Pruning','Correct / 10,000','Accuracy (%)','CSR payload (bytes)','Dense MACs / image'],[[f"{v['rate']*100:g}% global L1",v['correct'],pct(v),v['sparse_payload_bytes'],v['dense_macs']] for v in pr['rates']]+[['75% + 3 recovery epochs',pr['recovered']['correct'],pct(pr['recovered']),pr['recovered']['sparse_payload_bytes'],416520],['Compacted + 3 recovery epochs',pr['compact']['after_recovery']['correct'],pct(pr['compact']['after_recovery']),f"{compact['dense_fp32_bytes']:,} dense FP32",compact['dense_macs']]])
    latency=table(['Model','Median (µs/image)','P90 (µs/image)'],[[v['model'],f"{v['median_us']:.2f}",f"{v['p90_us']:.2f}"] for v in la['results']])
    layer_table=table(['Layer','Output shape','Parameters','Dense MACs'],[['Input','1 × 32 × 32',0,0],['C1: Conv 5×5 + tanh','6 × 28 × 28',156,117600],['S2: AvgPool 2×2 + tanh','6 × 14 × 14',0,0],['C3: Conv 5×5 + tanh','16 × 10 × 10',2416,240000],['S4: AvgPool 2×2 + tanh','16 × 5 × 5',0,0],['C5: Conv 5×5 + tanh','120 × 1 × 1',48120,48000],['F6: Linear + tanh','84',10164,10080],['F7: Linear logits','10',850,840],['Total','—',61706,416520]])
    readme=f'''# LeNet Precision Lab: Quantization, Pruning and Hardware Weight Export

**Matin Firoozbakht** | [@matinfirooz](https://github.com/matinfirooz)

A complete Fashion-MNIST study with a trained **61,706-parameter LeNet-5**, manual NumPy quantization, PyTorch post-training quantization, pruning and hardware-ready weight images. Every accuracy uses the **full 10,000-image official test set**. The executed notebook contains the complete answers, source listings, measured tables and figures.

![LeNet architecture](hardware/figures/lenet_architecture.png)

## Included work

- **Manual implementation:** symmetric, integer-zero-point affine and the supplied shifted min–max formula at INT16, INT8 and INT4; complete LeNet inference uses NumPy only.
- **PyTorch implementation:** FP32, true CPU FP16, INT16/INT8/INT4/INT2 grids, frozen PTQ observers and both weight-only and activation quantization studies.
- **Native INT8:** five converted quantized convolution/linear operators, saved checkpoint and separate measured accuracy.
- **Pruning:** five fixed sparsity levels, mask-preserving recovery, CSR exports and actual channel compaction.
- **Hardware exports:** all five learned layers, binary/hex packed codes, biases, scales, zero-points, shapes, offsets and widened integer MAC golden vectors.
- **Verification:** {n_tests} tests, saved full-test predictions, model reload checks, checksummed data and source/result evidence.
- **Figures:** four hardware-style PNG/SVG diagrams and seven measured plot pairs.

## Measured precision results

FP32: **{tr['test']['correct']}/10,000 = {pct(tr['test'])}%**. FP16: **{next(v['correct'] for v in pt['results'] if v['name']=='FP16')}/10,000 = {pct(next(v for v in pt['results'] if v['name']=='FP16'))}%**. Native INT8: **{pt['native_int8']['correct']}/10,000 = {pct(pt['native_int8'])}%**.

{precision}

**WbA32** quantizes weights to b bits and retains FP32 activations and biases. **WbAb** additionally quantizes the input, tanh/pooling outputs and final logits; its arithmetic and nonlinear functions still execute in FP32. INT16/INT4/INT2 are quantization/dequantization (**Q/DQ**) simulations. The INT8 rows above also use the controlled Q/DQ grid; the native INT8 backend is reported separately because it uses per-channel weight scales and different operator quantization rules. Symmetric INT2 uses **three active values (−1, 0, 1)** in a two-bit container.

Storage is actual packed weight bytes plus retained biases and five 20-byte binary grid records. WbAb also stores eight activation grid records (**160 additional bytes**). JSON descriptions, NPZ/PT serialization, optional integer-bias references and intermediate activation buffers are outside this parameter-payload metric. FP16 stores all parameters as FP16. A Q/DQ checkpoint itself contains floating-point tensors; the packed export provides the compressed representation.

![PyTorch accuracy](results/figures/pytorch_accuracy.png)

## Complete deliverables

- [Executed Jupyter notebook](HW4_Matin_Firoozbakht.ipynb)
- [Complete written answers](COMPLETE_ANSWERS.md)
- [Manual implementation and results](software/quantization.py)
- [Hardware formats and layer contracts](hardware/README.md)
- [Figure catalog](hardware/FIGURES.md)
- [Requirements and interpretation](docs/SPECIFICATION.md)
- [Measured evidence](results/evidence.json)

## Run the notebook

Python **3.12** is the tested environment. Extract the complete ZIP and run these commands from its project directory:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
jupyter lab HW4_Matin_Firoozbakht.ipynb
```

Choose **Restart Kernel and Run All Cells**. The default notebook verifies saved evidence, recomputes FP32 evaluation, runs the worked example, reloads exported weights and displays all recorded experiments. Saved outputs are already embedded for reading without execution. Keep the notebook inside the extracted project when running it. The dataset and checkpoints are included; evaluation needs no data download.

Set `RECOMPUTE_EXPERIMENTS=True` to repeat the quantization/pruning study. Set `RETRAIN=True` as well to train FP32 again. These switches run the complete ordered reproduction before rebuilding the evidence. Training is deliberately separate from PTQ.

## Reproduce each stage

```bash
python scripts/train.py
python scripts/run_manual.py
python scripts/run_pytorch.py
python scripts/run_pruning.py
python scripts/benchmark.py
python scripts/qualify_packed.py
python -m unittest discover -s tests -v
python scripts/render_figures.py
python scripts/verify_evidence.py --record
python scripts/write_documentation.py
python scripts/build_notebook.py
python scripts/execute_notebook.py
```

For the complete ordered workflow, use `python scripts/run_all.py --train`. To repeat experiments using the saved FP32 model, use `python scripts/run_all.py`.

The manual stage can run in a NumPy-only environment using `requirements-manual.txt`; it loads the supplied trained NPZ and imports no neural-network package. PyTorch is used for training and Step 2.

## Data, training and calibration

Fashion-MNIST supplies 60,000 official training and 10,000 official test images. The deterministic class-wise split reserves **54,000 training / 6,000 validation** images. Calibration uses **1,000 training images**, seed 101, with no validation/test examples. All images are normalized by 255 and padded with a two-pixel zero border to 32×32.

The network uses dense C3 connectivity, tanh activations, average pooling and ten raw output logits. The selected checkpoint is epoch **{tr['best_epoch']}** of **{tr['epochs_completed']}**, chosen by validation cross-entropy. Training uses AdamW, batch 256, initial learning rate 0.001 and a fixed schedule. PTQ performs no parameter retraining. The separate pruning recovery experiments use three epochs and validation-only checkpoint selection. Test labels are used only for reported evaluation.

## Manual quantization results

{manual}

The supplied asymmetric example contains an arithmetic error: **22.99 maps to 197**, using the full-precision scale 88.14/255. The shifted-endpoint formula reconstructs code 130 as approximately **0.00411765**; an integer-zero-point affine grid reconstructs that code as exactly zero. Both formulations are implemented and distinguished throughout the notebook.

![Manual accuracy](results/figures/manual_accuracy.png)

## Pruning and compaction

{pruning}

Global zero masks retain the original tensor dimensions and dense MAC count. CSR storage includes FP32 values, uint16 columns, uint32 row pointers and FP32 biases. The compact model physically reduces channels to **{' / '.join(map(str,compact['channels']))}**, giving **{compact['parameters']:,} parameters** and **{compact['dense_macs']:,} dense MACs/image**. Its accuracy before recovery was {pct(pr['compact']['before_recovery'])}%; after three recovery epochs it is {pct(pr['compact']['after_recovery'])}%. The additional recovered-75%-sparse plus weight-only INT8 experiment scores {pct(pr['joint_weight_int8'])}%.

![Pruning](results/figures/pruning.png)

## Hardware weight images

![Packed weights](hardware/figures/packed_weights.png)

`hardware/exports/` contains FP32 and FP16 parameter images; symmetric and affine INT16/8/4/2 codes; native per-channel INT8 weights; and CSR pruning exports. Convolution layout is **OIHW**, linear layout **OI**. Codes are packed in C-order, starting with the least significant bits of the first byte. Signed codes use two’s complement. INT16 and all multi-byte fields are little-endian.

`metadata.json` records exact shapes, offsets, numerical formats and checksums. Every custom-grid export reloads to the evaluated Q/DQ weights exactly. The C1 golden example checks an exact widened integer dot product with explicit zero-point correction. Integer bias references and accumulator bounds support implementation planning. See [hardware/README.md](hardware/README.md) for formulas and decoder examples.

The diagrams describe the model, binary layouts and an arithmetic mapping reference. This project implements the software study and hardware data exports; full-network RTL, synthesis, routed timing and power are not measured deliverables.

## Run inference from the packed weights

```bash
python scripts/infer_packed.py --method affine --bits 8 --index 0
python scripts/infer_packed.py --method symmetric --bits 4 --mode weights_activations --all
python scripts/qualify_packed.py
```

The NumPy runtime reads packed weights, FP32 biases and binary grid records directly; it does not load a PT/NPZ model or import PyTorch. All **{len(packed['results'])}** custom-grid configurations were evaluated on the full test set. Total prediction differences from their saved PyTorch Q/DQ references: **{sum(v['saved_torch_prediction_disagreements'] for v in packed['results'])}**. This is a test of the exported parameters and NumPy Q/DQ model; it is not an integer-only accelerator timing result.

## CPU measurements

{latency}

Measured on **{la['cpu']}**, one CPU thread, batch 1, {la['warmup_calls']} warmup calls and {la['measured_calls']} timed calls per model. Python dispatch is included. These shared-CPU measurements do not establish a general quantization or sparsity speedup. Lower precision reduces the exported parameter payload; latency also depends on backend, shapes and dispatch overhead.

## Files

| Path | Contents |
| --- | --- |
| `software/` | Model, data loader, manual quantization/inference, PyTorch PTQ, pruning, exports, figures and reports |
| `models/` | FP16, custom-grid Q/DQ, native INT8, sparse and compact checkpoints |
| `hardware/exports/` | Hardware-ready binary and hex parameter images with exact metadata |
| `hardware/figures/` | Four architecture/datapath/layout/pruning PNG/SVG pairs |
| `results/quantization/` | 34 quantization runs, 16 packed-model runs, calibration ranges, examples and predictions |
| `results/pruning/` | Pruning, recovery, compact-model measurements and predictions |
| `results/software/` | Training, data splits, calibration indices, FP32 outputs and CPU latency |
| `results/figures/` | Seven measured plot PNG/SVG pairs |
| `scripts/`, `tests/` | Ordered reproduction, notebook execution and {n_tests} tests |
| `data/fashion/`, `docs/` | Checksummed official data, dataset license and supplied requirements |

Dataset licensing is retained in `data/fashion/LICENSE`. Authored source is provided under the [MIT license](LICENSE). Full implementation choices, quantitative answers and primary references are in [COMPLETE_ANSWERS.md](COMPLETE_ANSWERS.md).
'''
    (r/'README.md').write_text(readme)
    examples=load(r,'results/quantization/worked_example.json')
    sampletable=table(['Method','Bits','Scale','Zero-point','MSE','Max absolute error'],[[v['method'],v['bits'],f"{v['scale']:.10g}",v['zero_point'],f"{v['mse']:.8g}",f"{v['max_error']:.8g}"] for v in examples])
    answer=r'''# Complete Answers: LeNet Quantization and Pruning

**Matin Firoozbakht** | [@matinfirooz](https://github.com/matinfirooz)

## 1. Purpose and memory calculation

Quantization represents continuous parameters and activations using a finite grid. The main direct benefit in this project is a smaller parameter payload. Native integer kernels can also change execution cost, while Q/DQ simulation measures numerical effects with floating-point operators. Pruning removes selected connections or channels; a suitable sparse format or actual shape reduction is needed to turn zeros into storage or compute reductions.

For 175 billion parameters at 32 bits:

$$M = 175\times10^9\times32/8 = 700\times10^9\ \text{bytes}.$$

This is **700 GB decimal**, approximately **651.93 GiB**, before activation buffers, optimizer state and runtime metadata. Ideal FP16 and INT8 parameter-only storage would be 350 GB and 175 GB, respectively. Scales, zero-points and bias policies add overhead.

## 2. Manual symmetric quantization

For b bits, the assignment uses a narrow signed range:

$$Q=2^{b-1}-1,\quad q_{\min}=-Q,\quad q_{\max}=Q,$$
$$s=\max_i |x_i|/Q,\quad z=0,$$
$$q=\operatorname{clip}(\operatorname{round}(x/s),-Q,Q),\quad \hat{x}=sq.$$

All-zero tensors use scale 1 and code 0. This implementation uses nearest-even rounding and saturates explicitly. In the absence of clipping, nearest-neighbor reconstruction error is at most half a scale step. The often-used estimate MSE ≈ s²/12 additionally assumes a roughly uniform quantization residual; it is not a guarantee for these weights.

At two bits the narrow symmetric range is **−1 to +1**, so it has three active levels. A two-bit container has four bit patterns; the −2 code is unused. This distinction matters when interpreting the very poor two-bit results.

## 3. Two asymmetric formulations

### 3.1 Literal shifted endpoint formula

The supplied PDF states:

$$s=(x_{\max}-x_{\min})/(2^b-1),$$
$$q=\operatorname{clip}(\operatorname{round}((x-x_{\min})/s),0,2^b-1),$$
$$\hat{x}=sq+x_{\min}.$$

This uses a real-valued offset. It preserves the two endpoints but generally cannot make a floating-point zero exactly representable with an integer zero-point. Constant tensors use scale 1, code 0 and their constant value as the offset.

### 3.2 Hardware affine grid with an integer zero-point

Use a range that includes zero:

$$l=\min(x_{\min},0),\quad h=\max(x_{\max},0),$$
$$s=(h-l)/(2^b-1),\quad z=\operatorname{clip}(\operatorname{round}(-l/s),0,2^b-1),$$
$$q=\operatorname{clip}(\operatorname{round}(x/s)+z,0,2^b-1),\quad \hat{x}=s(q-z).$$

All-zero tensors use s=1 and z=0. Positive and negative constant tensors are handled by including zero in the calibration range. In this formulation code z reconstructs exactly to zero, which is useful for padding, sparse zeros and integer MAC correction. Symmetric and affine methods are compared under the same per-tensor policy; the native backend is a separate per-channel experiment.

### 3.3 Corrected worked example

Input:

```text
[43.21, -44.93, 0, 22.99, -43.93, -11.35, 38.48, -20.49, -38.61, -28.02]
```

Eight-bit symmetric scale: 44.93/127 = 0.3537795275590551.

```text
[122, -127, 0, 65, -124, -32, 109, -58, -109, -79]
```

Eight-bit asymmetric scale: 88.14/255 = 0.3456470588235294. Both asymmetric methods give the following integer codes for this sample:

```text
[255, 0, 130, 197, 3, 97, 241, 71, 18, 49]
```

The supplied 196 is incorrect: (22.99 + 44.93)/s rounds to **197**. In the endpoint grid, code 130 reconstructs to **0.00411764705882689**, while in the affine grid it reconstructs to **0**. Matching codes on this example do not make the two reconstruction formulas equivalent. The implementation keeps full precision when computing scales rather than rounding a printed scale.

'''+sampletable+r'''

![Worked sample](results/figures/worked_example.png)

## 4. Data and FP32 model training

Official Fashion-MNIST training data are split class-wise using seed 42: 54,000 train and 6,000 validation. The official 10,000-image test set remains separate. Exactly 1,000 calibration indices are sampled from the training indices using seed 101 and saved. No test labels select checkpoints, ranges, pruning rates or bit-widths.

Images are divided by 255 and zero-padded from 28×28 to 32×32. The concrete LeNet variant is defined below. C3 is fully connected across its input channels, pooling has no trainable parameters, and tanh follows both convolutions and pooling. It is a clearly specified dense LeNet variant rather than a claim to reproduce the original sparse C3 and trainable-subsampling network exactly.

'''+layer_table+r'''

For each convolution, the dense MAC count is Hout × Wout × Cout × Cin × Kh × Kw. Each linear layer uses Cin × Cout MACs. Bias additions, tanh, pooling and softmax are outside this MAC count. F7 returns raw logits for cross-entropy training. Softmax is only a presentation of class probabilities; it is not applied before the training loss. Prediction uses the largest logit, with the first class index selected on a tie.

AdamW uses batch size 256, learning rate 0.001, weight decay 0.0001 and learning-rate drops by 0.5 at epochs 10 and 16. Initialization seed is 7; shuffle seed is 42. All '''+str(tr['epochs_completed'])+r''' epochs are completed and minimum validation cross-entropy selects epoch '''+str(tr['best_epoch'])+r'''. The official-test result is **'''+pct(tr['test'])+r'''%**. CPU execution uses '''+str(tr['cpu_threads'])+r''' threads for training. Exact seeds, timing, versions, history, split indices and source hashes are saved.

![Training](results/figures/training.png)

## 5. Step 1: NumPy-only model evaluation

`software/quantization.py` implements the numerical grids without importing a neural-network framework. `software/numpy_lenet.py` implements im2col convolution, matrix multiplication, tanh, average pooling and stable cross-entropy using NumPy. `scripts/run_manual.py` is run in an isolated process and records that no PyTorch, TensorFlow, JAX or Keras module was imported.

For weight-only evaluation, all five weight tensors are quantized per tensor, then reconstructed, while biases and activations stay FP32. The WbAb extension also quantizes eight activation points using fixed training-calibration extrema: input, C1 tanh, S2 tanh, C3 tanh, S4 tanh, C5 tanh, F6 tanh and logits. Arithmetic and nonlinear functions are FP32 in both modes.

The NumPy FP32 implementation agrees with every one of the 10,000 PyTorch baseline predictions. Maximum logit discrepancy is '''+f"{ma['baseline']['torch_max_logit_error']:.8g}"+r''', consistent with different floating-point reduction orders. These are numerical equivalence results, not integer-kernel timing results.

'''+manual+r'''

![Manual evaluation](results/figures/manual_accuracy.png)

## 6. Step 2: PyTorch PTQ and precision coverage

Train FP32 once, then quantize without changing its learned parameters through optimization. The assignment defines PTQ as post-training conversion; its later wording about training at each precision is interpreted as preparing and evaluating each quantized model from the trained baseline. Quantization-aware training is a different method and is not silently substituted for PTQ. The pruning recovery experiments are explicitly separate.

FP16 evaluates an actual half-precision model and half-precision inputs on the CPU. For INT16/8/4/2, PyTorch 2.5.1 `FakeQuantize` and `MinMaxObserver` compute per-tensor grid parameters. Custom ranges are carried by qint8/quint8 for up to eight logical bits and qint32 for the 16-bit grid. The storage carrier does not imply an INT16 convolution kernel. Weight observers see the trained weights. Activation observers see frozen FP32 extrema measured on the 1,000 training calibration images; observers are disabled for evaluation.

Weights are reconstructed on the selected grid and retained in a saved FP32 Q/DQ state dictionary. Activation quantization uses `torch.fake_quantize_per_tensor_affine`. Codes for export are recovered from the actual PyTorch fake-quantized output, then reconstructed and compared exactly. This avoids assuming that float32 reciprocal multiplication at rounding boundaries is always identical to NumPy float64 division.

'''+precision+r'''

A real converted INT8 model uses QuantStub/DeQuantStub, min–max activation observers with 0..255 codes, per-output-channel symmetric weight observers and the x86 quantized engine. All three Conv2d and both Linear modules are verified as quantized operators. It scores **'''+pct(pt['native_int8'])+r'''%** ('''+str(pt['native_int8']['correct'])+r'''/10,000). Tanh and pooling follow the quantized backend rules. Its grids differ from the controlled per-tensor Q/DQ experiment, so its accuracy difference is not a pure test of integer arithmetic alone. Native INT16, INT4 and INT2 speedups are not asserted.

![Framework evaluation](results/figures/pytorch_accuracy.png)

## 7. Answers to the analysis questions

'''+analysis+r'''

![Storage trade-off](results/figures/compression_accuracy.png)

## 8. Pruning extension

The source title and introduction mention pruning but prescribe no pruning algorithm or rates. This additional study fixes global L1 weight sparsities of 0%, 25%, 50%, 75% and 90%, preserving biases. PyTorch masks remain attached during recovery, so masked connections cannot regrow. Mask removal occurs only after recovery when producing a normal state dictionary. Due to rounding of the requested number of weights, 25% means 15,368 of 61,470 weights and 75% means 46,102 weights; exact measured sparsities are saved.

'''+pruning+r'''

The recovered 75% sparse model reaches '''+pct(pr['recovered'])+r'''%; the joint recovered-sparse plus weight-only symmetric INT8 model reaches '''+pct(pr['joint_weight_int8'])+r'''%. Recovery uses three AdamW epochs at learning rate 0.0001 with fixed seed 17 and validation cross-entropy checkpoint selection.

A second experiment physically compacts channels from 6/16/120/84 to 4/12/90/63. L1 output-row scores select channels from the trained dense weights, and the kept indices are propagated through subsequent inputs, including both dense layers. Before recovery the compact model is '''+pct(compact['before_recovery'])+r'''%; after three recovery epochs it is '''+pct(compact['after_recovery'])+r'''%. The compact tensors have '''+f"{compact['parameters']:,}"+r''' parameters and '''+f"{compact['dense_macs']:,}"+r''' dense MACs per image, reductions of '''+f"{(1-compact['parameters']/tr['parameters'])*100:.2f}"+r'''% and '''+f"{(1-compact['dense_macs']/416520)*100:.2f}"+r'''%. Before recovery its outputs match a corresponding masked full-sized network within '''+f"{pr['compact']['masked_equivalence_max_error']:.8g}"+r''' maximum logit error.

Sparse export uses FP32 values (4 bytes), uint16 column indices (2 bytes), uint32 row pointers and FP32 biases. For the full network there are 236 output rows across the five learned layers and five extra terminal row-pointer entries, hence 964 row-pointer bytes. Total CSR payload is **6 × nnz + 964 + 944 bytes**. Below sufficient sparsity, indices can make storage larger than dense FP32, as the 25% result demonstrates. Zero masks alone retain dense tensor dimensions and dense MAC counts. Channel compaction provides a direct reduction in those dimensions.

![Pruning](results/figures/pruning.png)

## 9. Hardware preparation and numerical contract

Convolutions use OIHW and linear layers OI layout, both flattened in C order. INT2/4/8/16 packing uses the first code in the least significant bits of the byte; signed values are two’s-complement and 16-bit words are little-endian. Each layer starts on a byte boundary. All padding bits are zero. Metadata gives shapes, counts, byte offsets, scales, zero-points and SHA-256.

For integer affine operands, the exact widened accumulation is:

$$A_j=\sum_{i=0}^{K-1}(q_{x,i}-z_x)(q_{w,ji}-z_w)+b_{q,j},$$
$$b_{q,j}=\operatorname{round}(b_j/(s_xs_w)),\qquad \hat{y}_j=s_xs_w A_j.$$

Equivalent correction before adding bias is:

$$\sum_i q_{x,i}q_{w,ji}-z_w\sum_i q_{x,i}-z_x\sum_i q_{w,ji}+Kz_xz_w.$$

The exporter supplies optional INT64 bias codes, per-layer conservative accumulator bounds, required signed widths and an exact central C1 patch golden vector. The first image's padded input patch at row/column 12 is used, providing nontrivial pixel values. W16 products/accumulations can exceed 32 bits, so the reference widens before subtraction and multiplication. Zero-point subtraction can require b+1 signed bits for an unsigned b-bit code.

The evaluated Q/DQ paths retain FP32 biases; the rounded integer bias references are not substituted into those evaluations. A complete integer-only accelerator would additionally need a defined tanh approximation, average-pooling rounding, requantization policy, buffering, interface protocols and cycle-level verification. The supplied hardware diagrams are authored mapping references, and no complete RTL, synthesis or routed PPA is claimed.

`hardware/README.md` specifies each binary file and shows a decoder. Every packed weight export is round-trip checked against the saved evaluated model, and the integer dot-product identity is tested independently. Native per-channel INT8 exports retain their own scales and zero-points rather than reusing the per-tensor grid metadata.



### 9.1 Executable packed-weight runtime

`software/packed_runtime.py` and `scripts/infer_packed.py` load the binary weight/bias/grid files. Weight SHA-256, byte counts, shapes, offsets, grid descriptions, signed ranges and padding bits are checked. Activation grids are read from `activation_qparams.bin` when WbAb is selected. No trained PT/NPZ checkpoint or neural-network framework is needed by this runtime.

```bash
python scripts/infer_packed.py --method affine --bits 8 --index 0
python scripts/infer_packed.py --method symmetric --bits 4 --mode weights_activations --all
```

`python scripts/qualify_packed.py` reproduces all sixteen full-test evaluations below. Differences are counted against the corresponding saved PyTorch model predictions rather than assumed to be zero. Arithmetic after decoding is NumPy FP32 Q/DQ; integer code packing does not turn this execution into native integer convolution.

'''+packed_table+r'''

## 10. CPU measurements

'''+latency+r'''

Host: **'''+la['cpu']+r'''**. PyTorch '''+la['torch_version']+r''', '''+la['engine']+r''' quantized engine, '''+str(la['threads'])+r''' thread, batch 1, '''+str(la['warmup_calls'])+r''' warmup and '''+str(la['measured_calls'])+r''' timed forwards. Median and P90 include Python dispatch on a shared CPU. These results are specific to this host and do not establish hardware energy, FPGA timing or a stable broad speedup. The lower dense MAC count of the compact model and the smaller packed payload are independent, directly measured structural properties.

![Latency](results/figures/latency.png)

## 11. Verification, reproducibility and limitations

'''+str(n_tests)+r''' unit/integration tests cover exact example arrays, nearest-even ties, saturation, degenerate ranges, invalid inputs, zero representation, every code value at each packing width, odd-length packing, endianness, zero-point correction, data checksums and disjoint calibration, parameter/MAC counts, NumPy/PyTorch parity, all sixteen saved Q/DQ model reloads, converted INT8 operators, mask persistence, sparse round trips, channel compaction and full-test prediction counts. Added checks exercise the packed-model CLI, reject damaged binary grids/payloads, detect newly added source files and verify that report text follows changed measurements.

`results/evidence.json` binds computational sources, original compressed data, checkpoints, hardware exports and numeric measurements to SHA-256 digests. `scripts/verify_evidence.py` detects modified/missing inputs and newly added computational files; `--record` records completed experiments and must be used only after rerunning the affected computation. Generated prose and figures are outside this numeric-evidence hash scope. The notebook displays saved results after checking evidence and independently recomputes the FP32 test accuracy; its optional switches repeat full experiments.

Results describe one trained seed and a fixed calibration sample. There are no confidence intervals across independently trained models. Per-tensor min–max PTQ is intentionally simple; QAT, percentile clipping and learned scales are possible follow-up methods, not measured results here. FP32/F16/native INT8 are actual backend executions; other integer precisions describe simulated grids and real packed storage. All sources and checkpoints are included, and no runtime download is needed for evaluation.

## 12. Primary references and supplied materials

1. Supplied [HW4.pdf](docs/HW4.pdf) and [original assignment README](docs/assignment_README.md).
2. Fashion-MNIST authors' [repository, dataset and license](https://github.com/zalandoresearch/fashion-mnist).
3. PyTorch 2.5.1 [FakeQuantize source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/ao/quantization/fake_quantize.py) and [observer source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/ao/quantization/observer.py). The pinned runtime is tested directly; current documentation may describe newer APIs.
4. PyTorch [official pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html).
5. M. Horowitz, “Computing’s energy problem (and what we can do about it),” ISSCC 2014, [DOI:10.1109/ISSCC.2014.6757323](https://doi.org/10.1109/ISSCC.2014.6757323). Historical context, not a PPA measurement of this project.
6. A. Gholami et al., [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630), 2021. Background reference from the supplied assignment.
'''
    # Markdown mathematics is kept GitHub/Jupyter compatible.
    answer=answer.replace('\\\\','\\')
    (r/'COMPLETE_ANSWERS.md').write_text(answer)
    specs='''# Requirement coverage and interpretation

Author: Matin Firoozbakht · https://github.com/matinfirooz

The original supplied PDF and README are preserved unchanged in this directory.

'''+table(['Requirement','Implementation / evidence'],[['Manual symmetric and asymmetric quantization','software/quantization.py; endpoint and integer-affine definitions kept separate'],['Provided sample','worked_example.json and corrected 197 code in the notebook'],['Manual INT16 / INT8 / INT4 accuracy','18 NumPy-only full-test sweeps, two activation policies and three grids'],['Accuracy-versus-bit plot','manual_accuracy.png and pytorch_accuracy.png'],['Train LeNet-5 on Fashion-MNIST',''+str(tr['epochs_completed'])+' actual epochs; 54k/6k split; official 10k test'],['FP32 / FP16 / INT16 / INT8 / INT4 / INT2','Saved checkpoints, full-test predictions and packed parameter images'],['Use PyTorch quantization tools','FakeQuantize/MinMaxObserver and a separate prepare/convert native INT8 path'],['Extract weights for hardware','All five learned layers exported with numerical and layout metadata'],['Trade-off and degradation questions','COMPLETE_ANSWERS.md sections 7–10; notebook sections 8–13'],['Pruning title/introduction','Explicit additional magnitude-pruning and channel-compaction study'],['Submission identity','Matin Firoozbakht; GitHub @matinfirooz']])+'''

The source defines PTQ as conversion after FP32 training but later uses wording that could be read as training after conversion. This solution trains FP32, then evaluates PTQ without retraining. Recovery of pruned models is separately identified and never described as pure PTQ. No particular pruning rates or algorithms are specified; extension choices are fixed and reported.

The PDF does not prescribe an exact LeNet layer table, preprocessing policy or calibration size. The implemented dense C3/tanh/average-pooling variant, normalization, padding and splits are fully documented. Training/checkpoint selection and calibration do not use official test labels. The source's shifted min–max and integer-zero-point descriptions are numerically distinct; both are implemented and the sample arithmetic is corrected.

No student number or separate official report template was supplied. The named ZIP/notebook use HW4 and the author's full name. Transfer content to an official template and add a student number if the course separately requires them.
'''
    (r/'docs/SPECIFICATION.md').write_text(specs)
    q=load(r,'hardware/exports/affine_int8/metadata.json')
    bounds=table(['Layer','K','Input scale','Weight scale','Bias scale','Required signed accumulator bits'],[[a['layer'],int(__import__('math').prod(a['shape'][1:])),f"{a['input_qparams']['scale']:.8g}",f"{a['qparams']['scale']:.8g}",f"{a['bias_scale']:.8g}",a['required_signed_accumulator_bits']] for a in q['layers']])
    guide='''# Hardware Parameter Export Guide

**Matin Firoozbakht** | [@matinfirooz](https://github.com/matinfirooz)

This package prepares exact numerical parameter images for implementing the trained LeNet. It supplies export/decoder software and integer dot-product references. The diagrams illustrate hardware mappings; complete accelerator RTL and physical implementation are outside the measured scope.

![Datapath](figures/quantized_datapath.png)

## Network and layer shapes

'''+layer_table+'''

## Export directories

| Directory | Contents |
| --- | --- |
| `fp32/`, `fp16/` | IEEE parameter images, all weights and biases, exact tensor offsets |
| `symmetric_int{16,8,4,2}/` | Signed narrow symmetric packed codes and scale records |
| `affine_int{16,8,4,2}/` | Unsigned affine packed codes with integer zero-points |
| `native_int8/` | Converted CPU operator weights, per-output-channel scales/zeros and biases |
| `pruned_{0,25,50,75,90}_csr/` | Sparse FP32 values, columns, row pointers and biases |
| `pruned_75_recovered_csr/` | CSR after three mask-preserving recovery epochs |

## Custom-grid files

| File | Exact contract |
| --- | --- |
| `weights.bin` | Packed C-order OIHW/OI weights; each layer starts on a byte boundary |
| `weights.hex` | Same bytes, two hex digits per line, suitable for byte-wide ROM loading |
| `biases_fp32.bin` | Little-endian FP32 biases in layer order |
| `qparams.bin` | Five records: little-endian float64 scale, int32 zero, float64 offset; 20 bytes each |
| `activation_qparams.bin` | Eight 20-byte records in metadata activation-point order |
| `codes.npz` | Named integer arrays for inspection; not a compressed-payload benchmark |
| `{layer}_bias_i64.bin` | Rounded bias/(input_scale × weight_scale), little-endian signed INT64 |
| `conv1_integer_golden.npz` | Central C1 patch codes, weights, rounded bias, exact INT64 accumulators and scaled outputs |
| `metadata.json` | Shapes, counts, offsets, bit ranges, scales, accumulator bounds and weight digest |

Code 0 occupies the least significant bits of byte 0. INT2 packs four codes/byte; INT4 packs two; INT8 packs one; INT16 uses a little-endian two-byte word. Signed codes are two's complement. The unused narrow symmetric minimum (−2 at INT2, −8 at INT4, etc.) is not emitted. Tail padding bits are zero. For example, unsigned INT4 codes [1,2,3] encode as bytes [0x21,0x03].

A symmetric weight reconstructs as q×scale; an affine weight as (q−zero)×scale. The offset field is zero for these Step 2 exports and is reserved for the separate shifted-endpoint formulation. Both retained FP32 bias and optional integer-bias reference are included, with their roles identified rather than combined in reported payload size.

```python
from pathlib import Path
from software.exports import load_export
weights, metadata = load_export(Path('hardware/exports/affine_int8'))
print(weights['conv3.weight'].shape)  # (16, 6, 5, 5)
print(metadata['sha256'])
```

`load_export` verifies the weight digest and payload lengths, reads scales/zero-points from the binary grid records, checks their metadata descriptions, decodes sign and sub-byte packing and reconstructs every tensor. Tests compare these tensors exactly with all evaluated custom-grid checkpoints.

## Execute the exported model

```bash
python scripts/infer_packed.py --method affine --bits 8 --index 0
python scripts/infer_packed.py --method affine --bits 8 --mode weights_activations --all
python scripts/qualify_packed.py
```

The runtime loads the binary parameters and uses the NumPy model. It reports class probabilities for one image or full-test accuracy. Full qualification checks all 16 custom-grid configurations against the saved PyTorch Q/DQ predictions. `results/quantization/packed_results.json` contains actual counts, logit discrepancies and parameter-file hashes. No PT/NPZ checkpoint or PyTorch import is required for packed inference. These are Q/DQ evaluations, not integer-only kernel timings.

## Integer MAC, bias and bounds

Compute `sum((qx-zx)*(qw-zw)) + bias_i64` after widening both code operands before subtraction. Equivalent expansion is `sum(qx*qw) - zw*sum(qx) - zx*sum(qw) + K*zx*zw + bias_i64`. The reference uses INT64 throughout; multiply by input_scale×weight_scale to reconstruct a preactivation. Rounding integer biases introduces a separate error of up to half that product scale; the evaluated Q/DQ models retain FP32 bias.

The conservative absolute bound is K×max|qx−zx|×max|qw−zw| + max|bias_code|. Required signed bits are computed from this bound, with no assumption that an INT32 accumulator is adequate at 16-bit operands. Example, affine W8A8:

'''+bounds+'''

The golden patch is the first official test image's padded input window [12:17,12:17] for all six C1 filters. The direct centered dot product and expanded zero-point correction must agree exactly. This checks integer arithmetic for a nontrivial local operation; it is not a claim that the full-network Q/DQ results are integer-only accelerator results.

A full deployment must define nonlinear-function accuracy, pooling rounding, requantization multipliers and shifts, saturation, SRAM banking, dataflow, interfaces and clock constraints. The authored datapath shows an init mux loading bias into an accumulator and subsequent multiply-accumulate feedback; it is an arithmetic mapping reference.

## Native INT8 files

The native backend uses separate `{layer}_weight.bin` int8 codes, `{layer}_scale.bin` float64 scales, `{layer}_zero.bin` int32 zero-points and `{layer}_bias.bin` FP32 biases. Scales/zero-points are per output channel (axis 0). `activation_qparams.bin` has six little-endian float64-scale/int32-zero records: input then five learned-layer outputs. Tanh/average-pool quantization follows the pinned PyTorch backend. Its exported metadata must not be replaced with custom per-tensor grid records.

## Sparse weights and compaction

CSR flattens each convolution filter's C,H,W dimension. Each layer stores FP32 values, uint16 columns, uint32 row pointers and FP32 biases, all little-endian. A row contains entries between `rowptr[r]` and `rowptr[r+1]`; missing coordinates are zero. The maximum flattened input width is 400, so uint16 columns are sufficient. Dense execution of the sparse checkpoints still performs all original MACs. The compact checkpoint has genuinely smaller dimensions and 231,700 dense MACs/image.

## Storage accounting

WbA32 = packed weights + 944 FP32 bias bytes + 100 weight-grid bytes. At INT2 the C1 tail byte has four zero padding bits; all other layer payloads are byte-aligned. WbAb adds 160 activation-grid bytes. Optional INT64 bias files, golden vectors, NPZ/PT files and JSON are separately identified auxiliary artifacts. Native INT8 payload counts all exported codes/scales/zero-points/biases plus its 72 activation-grid bytes. Working activation memories, instruction/control state and implementation-specific alignment are outside these parameter-only sizes.

## Figures

See [FIGURES.md](FIGURES.md) for PNG and SVG links. All four diagrams use explicit direction arrows, orthogonal routes, memory/register symbols and typed numerical contracts where appropriate. They are authored illustrations, not Vivado or synthesis screenshots.
'''
    (r/'hardware/README.md').write_text(guide)
    catalog='# Hardware and Experiment Figures\n\n**Matin Firoozbakht** · [@matinfirooz](https://github.com/matinfirooz)\n\n'
    for n,title in [('lenet_architecture','LeNet layers and precision boundaries'),('quantized_datapath','Affine correction, init mux and accumulator feedback'),('packed_weights','Packed weight bytes and decoder'),('pruning_compaction','Sparse masks and channel compaction')]:catalog+=f'## {title}\n\n[SVG](figures/{n}.svg) · [PNG](figures/{n}.png)\n\n![{title}](figures/{n}.png)\n\n'
    catalog+='## Measured plots\n\n'+table(['Plot','PNG','SVG'],[[n,f'[PNG](../results/figures/{n}.png)',f'[SVG](../results/figures/{n}.svg)'] for n in ['training','manual_accuracy','pytorch_accuracy','compression_accuracy','pruning','worked_example','latency']])+'\n'
    (r/'hardware/FIGURES.md').write_text(catalog)
    for name,data in [('manual_accuracy',ma['results']),('pytorch_accuracy',pt['results']),('pruning',pr['rates'])]:
        path=r/f'results/{name}.csv';keys=list(dict.fromkeys(k for v in data for k in v))
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
    return {'manual':manual,'precision':precision,'pruning':pruning,'latency':latency,'layers':layer_table}
