from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from software.figures import main
main(ROOT)
print('Saved all plots and hardware mapping figures as PNG and SVG.')
