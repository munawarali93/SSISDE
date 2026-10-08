"""Execute the complete experiment in a fresh kernel of this Python environment."""
import os
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('notebook', nargs='?', default='experiments/Stochastic_SINDy_Black_Scholes.ipynb')
parser.add_argument('--in-place', action='store_true', help='Save completed outputs in the experiment notebook.')
args = parser.parse_args()
path = (root / args.notebook).resolve()
output = root / 'outputs' / path.parent.relative_to(root / 'experiments')
output.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
os.environ.setdefault('IPYTHONDIR', str(output / '.ipython'))
os.environ.setdefault('JUPYTER_PLATFORM_DIRS', '1')
name = path.name
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)

def report(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        first_line = cell.source.splitlines()[0] if cell.source else '(empty)'
        print(f'Running cell {cell_index}: {first_line}', flush=True)

client = NotebookClient(
    notebook,
    timeout=3600,
    kernel_name='python3',
    resources={'metadata': {'path': str(path.parent)}},
    on_cell_start=report,
)
try:
    client.execute()
finally:
    destination = output / name.replace('.ipynb', '.executed.ipynb')
    nbformat.write(notebook, destination)
    print(f'Execution record: {destination}', flush=True)
if args.in_place:
    nbformat.write(notebook, path)
print('PASS: full experiment completed.', flush=True)
