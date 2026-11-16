"""Ordered reproduction: no implicit retraining, flags make long work explicit."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--epochs',type=int,default=20);args=p.parse_args()
def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
if args.train:run('scripts/train.py','--epochs',str(args.epochs))
for script in ['run_manual.py','run_pytorch.py','run_pruning.py','benchmark.py']:run('scripts/'+script)
run('scripts/infer_packed.py','--all','--output','results/quantization/packed_affine_int8_weights_only.json')
run('scripts/qualify_packed.py')
run('-m','unittest','discover','-s','tests','-v')
run('scripts/render_figures.py');run('scripts/verify_evidence.py','--record');run('scripts/write_documentation.py');run('scripts/build_notebook.py');run('scripts/execute_notebook.py')
