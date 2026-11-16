from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from software.reporting import generate
generate(ROOT)
print('Generated README, complete answers, requirement coverage, hardware guide and result tables.')
