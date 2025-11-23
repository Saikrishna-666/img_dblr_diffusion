import torch
from pathlib import Path
import sys
repo_dir = Path(__file__).resolve().parents[1]
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from models.MRDNet import MRDNet

print('Using Python', sys.executable)
print('Torch version', torch.__version__)

# instantiate model with DFD enabled
m = MRDNet(use_dfd=True)
print('Model created. use_dfd:', getattr(m, 'use_dfd', False))

# run on CPU to avoid CUDA requirements
device = torch.device('cpu')
m = m.to(device)

# dummy input: batch 1, 3x256x256
x = torch.randn(1, 3, 256, 256, device=device)

try:
    outs = m(x)
    print('Forward pass succeeded.')
    if isinstance(outs, (list, tuple)):
        print('Number of outputs:', len(outs))
        for i, o in enumerate(outs):
            print(f'out[{i}] shape: {tuple(o.shape)}')
    else:
        print('Single output shape:', tuple(outs.shape))
except Exception as e:
    print('Forward pass failed with exception:')
    import traceback
    traceback.print_exc()

ld = getattr(m, 'last_dfd_maps', None)
print('last_dfd_maps is None?', ld is None)
if ld:
    for name, t in zip(['half','quarter'], ld):
        if isinstance(t, torch.Tensor):
            print(f'DFD {name} shape: {tuple(t.shape)}')
        else:
            print(f'DFD {name} is not a tensor:', type(t))

for attr in ['DFD_half','DFD_quarter']:
    mod = getattr(m, attr, None)
    if mod is not None:
        la = getattr(mod, 'last_attn', None)
        print(f'{attr}.last_attn is None? {la is None}')
        if la is not None:
            print(f'{attr}.last_attn shape: {tuple(la.shape)}')

print('Smoke test finished')
