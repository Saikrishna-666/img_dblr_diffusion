import torch
from pathlib import Path
import sys
repo_dir = Path(__file__).resolve().parents[1]
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))
from models.MRDNet import MRDNet

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    n = torch.cuda.device_count()
    print("GPU count:", n)
    for i in range(n):
        print(i, torch.cuda.get_device_name(i))

    use_dfd = True
    model = MRDNet(use_dfd=use_dfd)
    if torch.cuda.is_available():
        if n > 1:
            print(f"Wrapping model with DataParallel over {n} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.cuda()

    # Dummy forward to ensure scattering works across devices
    x = torch.randn(2 * max(1, n), 3, 256, 256)  # make batch >= num_gpus
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        y = model(x)
    print("Forward OK. Output shapes:", [t.shape for t in y])
