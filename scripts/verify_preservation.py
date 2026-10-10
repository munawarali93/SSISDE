"""Verify that the reorganized experiment preserves every scientific source cell."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
name = 'Stochastic_SINDy_Black_Scholes.ipynb'
original = json.loads((root / 'original' / name).read_text())
path = root / 'experiments' / 'black_scholes' / name
experiment = json.loads(path.read_text())
seen = []
for cell in experiment['cells']:
    index = cell.get('metadata', {}).get('original_cell_index')
    if index is None:
        continue
    seen.append(index)
    # Installation is replaced by kernel-specific requirements installation.
    if index == 1:
        continue
    reconstructed = []
    for line in ''.join(cell['source']).splitlines(keepends=True):
        if line.startswith('%run -i '):
            reconstructed.append((path.parent / line.removeprefix('%run -i ').strip()).read_text())
        else:
            reconstructed.append(line)
    expected = ''.join(original['cells'][index]['source'])
    if ''.join(reconstructed) != expected:
        raise AssertionError(f'Original cell {index} has changed')
if seen != list(range(len(original['cells']))):
    raise AssertionError('Original cells are missing, duplicated, or out of order')
print('PASS: all scientific cells and markdown preserved exactly, in original order.')
