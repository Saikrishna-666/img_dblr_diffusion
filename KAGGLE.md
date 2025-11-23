# Run MRDNet on Kaggle with 2× T4 GPUs

This guide shows how to train MRDNet (optionally with DFD) on Kaggle and verify it uses both T4 GPUs via DataParallel.

## Prerequisites
- Create a Kaggle Notebook.
- In Notebook Settings, set:
  - Accelerator: GPU
  - GPU Type: T4 x2 (Two T4)
  - Internet: On (to allow optional pip installs)

## Verify GPUs
Add a code cell:

```python
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

Expected: GPU count should be 2 and both devices are T4.

## Prepare repository
If your repo is in Kaggle Datasets, add it as input. Otherwise, clone:

```python
# Optional: clone your repo (requires Internet enabled)
!git clone https://github.com/Saikrishna-666/img_dblr.git -b main
%cd img_dblr/MRDNet
```

If you uploaded this folder as a Dataset input, cd into it instead:

```python
%cd /kaggle/input/your-dataset-name/MRDNet
```

## Install dependencies (optional)
Kaggle has recent PyTorch/torchvision pre-installed. If needed:

```python
!pip install --no-cache-dir -q tqdm scikit-image tensorboard
```

## Launch training on 2 GPUs
The code automatically uses nn.DataParallel when more than 1 GPU is available during training. You’ll see a message like:

```
Using 2 GPUs with DataParallel
```

Run training (edit data_dir to your dataset path):

```python
!python main.py \
  --mode train \
  --model_name MRDNet \
  --data_dir /kaggle/input/go-pro/GOPRO \
  --batch_size 8 \
  --num_epoch 2 \
  --use_dfd \
  --use_amp True \
  --print_freq 20
```

Tips:
- Increase batch_size to better utilize both GPUs (fits memory permitting).
- Mixed precision (AMP) is enabled by default when CUDA is available.

## Monitor GPU usage
Add a monitoring cell in another cell while training runs:

```python
!nvidia-smi -L
!watch -n 10 nvidia-smi
```

You should see both GPUs utilized by Python.

## Checkpoints and results
- Checkpoints: `results/MRDNet/weights/` (or `--checkpoint_dir` if provided)
- Images: `results/MRDNet/result_image/`

## Notes
- Multi-GPU path uses nn.DataParallel (simple and works well on Kaggle). For best scaling, consider PyTorch DistributedDataParallel in a script environment; Kaggle Notebooks also support `torchrun`, but DP is simpler.
- Device allocation bugs were fixed so wavelet ops allocate tensors on the correct device for multi-GPU.