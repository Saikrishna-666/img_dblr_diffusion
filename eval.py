import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from tqdm import tqdm


def _eval(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.test_model, map_location=device)
    # Accept formats: {'model': sd} or raw sd
    def _load_model(sd):
        try:
            model.load_state_dict(sd)
        except RuntimeError:
            from collections import OrderedDict
            new_sd = OrderedDict()
            first_key = next(iter(sd))
            if first_key.startswith('module.'):
                for k, v in sd.items():
                    new_sd[k.replace('module.', '', 1)] = v
            else:
                for k, v in sd.items():
                    new_sd['module.' + k] = v
            model.load_state_dict(new_sd)

    if isinstance(state, dict) and 'model' in state:
        _load_model(state['model'])
    elif isinstance(state, dict):
        _load_model(state)
    else:
        raise ValueError(f"Unrecognized checkpoint format at {args.test_model}")
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()  # pytorch中的显存机制
    adder = Adder()
    model.eval()  # 前向推理之前使用，防止test的batch_size过小，很容易被BN层影响结果
    with torch.no_grad():
        psnr_adder = Adder()

        # Hardware warm-up
        for iter_idx, data in enumerate(tqdm(dataloader, desc='Warm-up', leave=False)):
            input_img, label_img, _ = data
            input_img = input_img.to(device)
            tm = time.time()
            _ = model(input_img)
            _ = time.time() - tm

            if iter_idx == 20:
                break

        # Main Evaluation
        saved_count = 0
        save_limit = getattr(args, 'save_limit', 0) or 0
        for iter_idx, data in enumerate(tqdm(dataloader, desc='Evaluate', leave=False)):
            input_img, label_img, name = data

            input_img = input_img.to(device, non_blocking=True)

            tm = time.time()

            # Enable autocast for eval to reduce memory
            use_amp = torch.cuda.is_available() and getattr(args, 'use_amp', True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(input_img)[2]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image and (save_limit == 0 or saved_count < save_limit):
                save_name = os.path.join(args.result_dir, name[0])
                # Create subdirectories if name contains path segments (e.g., scene/filename.png)
                save_dir = os.path.dirname(save_name)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                pred_clip += 0.5 / 255
                pred_img = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred_img.save(save_name)
                saved_count += 1

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            # tqdm already shows progress; keep print for logs
            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))

            # Free per-iteration tensors after PSNR computation
            del pred, pred_clip, pred_numpy

    print('==========================================================')
    print('The average PSNR is %.2f dB' % (psnr_adder.average()))
    print("Average time: %f" % adder.average())
