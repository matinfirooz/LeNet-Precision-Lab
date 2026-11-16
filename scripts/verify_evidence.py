import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from software.evidence import snapshot,validate
p=argparse.ArgumentParser();p.add_argument('--record',action='store_true');args=p.parse_args()
if args.record:print('Recorded',snapshot(ROOT),'input/result hashes.')
else:
    r=validate(ROOT);print(json.dumps(r,indent=2));sys.exit(0 if r['status']=='passed' else 1)
