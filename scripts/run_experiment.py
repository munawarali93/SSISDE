"""Execute the complete experiment in a fresh kernel of this Python environment."""
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
output = root / 'outputs'
output.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
os.environ.setdefault('IPYTHONDIR', str(output / '.ipython'))
os.environ.setdefault('JUPYTER_PLATFORM_DIRS', '1')
name = 'Stochastic_SINDy_Black_Scholes.ipynb'
notebook = nbformat.read(root / 'experiments' / name, as_version=4)
nbformat.validate(notebook)

def report(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        first_line = cell.source.splitlines()[0] if cell.source else '(empty)'
        print(f'Running cell {cell_index}: {first_line}', flush=True)

client = NotebookClient(
    notebook,
    timeout=3600,
    kernel_name='python3',
    resources={'metadata': {'path': str(root / 'experiments')}},
    on_cell_start=report,
)
try:
    client.execute()
finally:
    destination = output / name.replace('.ipynb', '.executed.ipynb')
    nbformat.write(notebook, destination)
    print(f'Execution record: {destination}', flush=True)
print('PASS: full experiment completed.', flush=True)
