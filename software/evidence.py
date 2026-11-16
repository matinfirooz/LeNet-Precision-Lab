"""Bind saved results to their computational inputs. No library/framework imports."""
import hashlib,json
from pathlib import Path

def tracked(root):
    patterns=['software/*.py','scripts/*.py','tests/*.py','data/fashion/*.gz','software/*.npz','software/*.pt','models/*','hardware/exports/**/*','results/software/*.json','results/software/*.npz','results/software/*.npy','results/quantization/*.json','results/quantization/*.npz','results/pruning/*.json','results/pruning/*.npz']
    return sorted({p.relative_to(root).as_posix() for pattern in patterns for p in root.glob(pattern) if p.is_file() and p.suffix not in ['.pyc']})

def snapshot(root):
    root=Path(root);records={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in tracked(root)}
    result={'format_version':1,'scope':'Computational sources, official data, checkpoints, hardware exports and numeric measurements; generated narrative and figures excluded','files':records}
    (root/'results/evidence.json').write_text(json.dumps(result,indent=2));return len(records)

def validate(root):
    root=Path(root);data=json.loads((root/'results/evidence.json').read_text());bad=[]
    for n,digest in data['files'].items():
        p=root/n
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:bad.append(n)
    extra=sorted(set(tracked(root))-set(data['files']))
    return {'status':'passed' if not bad and not extra else 'stale','checked_files':len(data['files']),'changed_or_missing':bad,'unrecorded_files':extra}
