import os
import torch
from tqdm import tqdm

from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()  # 构造L1损失函数
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    dataloader = train_dataloader(
        args.data_dir,
        args.batch_size,
        args.num_worker,
        proportion=args.train_proportion,
        crop_size=args.crop_size
    )
    # Print number of training samples selected
    try:
        train_len = len(dataloader.dataset)
    except Exception:
        # Fallback if dataset is wrapped or non-standard
        train_len = sum(1 for _ in dataloader)
    print(f"Training samples: {train_len}")
    # print("dataloader=", dataloader)  # <torch.utils.data.dataloader.DataLoader object at 0x0000023BE5971AF0>
    max_iter = len(dataloader)  # 最大迭代次数是数据的长度,因为每次迭代四次
    # print("len(dataloader)=", max_iter)  # len(dataloader)= 526
    # 按需调整学习率，lr_steps是一个递增的list，gamma是学习率调整倍数，默认为0.1倍，这里根据参数设置为0.5，即下降50倍
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)

    epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        # Three possible formats:
        # 1) Full checkpoint dict: {'model': sd, 'optimizer': sd, 'scheduler': sd, 'epoch': int}
        # 2) Weights-only dict under 'model': {'model': sd}
        # 3) Raw state_dict: {...params...}

        def _load_state_into_model(sd: dict):
            """Load a state_dict into model with best-effort compatibility.
            - Handles DataParallel prefix differences.
            - Uses strict=False to allow architecture changes (e.g., enabling DFD later).
            - Prints missing/unexpected keys for transparency.
            """
            try:
                missing, unexpected = model.load_state_dict(sd, strict=False)
            except RuntimeError:
                # Try removing/adding 'module.' prefix
                from collections import OrderedDict
                new_sd = OrderedDict()
                first_key = next(iter(sd))
                if first_key.startswith('module.'):
                    for k, v in sd.items():
                        new_sd[k.replace('module.', '', 1)] = v
                else:
                    for k, v in sd.items():
                        new_sd['module.' + k] = v
                missing, unexpected = model.load_state_dict(new_sd, strict=False)
            # Log summary
            if missing:
                print(f"[Resume] Missing keys (initialized randomly): {len(missing)} e.g., {missing[:5]}")
            if unexpected:
                print(f"[Resume] Unexpected keys (ignored): {len(unexpected)} e.g., {unexpected[:5]}")

        if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
            _load_state_into_model(state['model'])
            restored = ['model']
            if 'optimizer' in state and isinstance(state['optimizer'], dict):
                try:
                    optimizer.load_state_dict(state['optimizer'])
                    restored.append('optimizer')
                except Exception:
                    pass
            if 'scheduler' in state and isinstance(state['scheduler'], dict):
                try:
                    scheduler.load_state_dict(state['scheduler'])
                    restored.append('scheduler')
                except Exception:
                    pass
            if 'epoch' in state and isinstance(state['epoch'], int):
                epoch = state['epoch'] + 1
                restored.append('epoch')
            else:
                epoch = max(1, int(getattr(args, 'start_epoch', 1)))
            print('Resume: restored ' + ', '.join(restored))
        elif isinstance(state, dict):
            # Assume it's a plain model state_dict
            _load_state_into_model(state)
            # Use explicit start_epoch if provided
            epoch = max(1, int(getattr(args, 'start_epoch', 1)))
            print('Resume: restored model (weights-only state_dict)')
        else:
            raise ValueError('Unrecognized checkpoint format for resume: %s' % type(state))

    writer = SummaryWriter()  # 实例化摘要和文件
    # Mixed precision scaler (enabled when CUDA is available and user allows AMP)
    use_amp = torch.cuda.is_available() and getattr(args, 'use_amp', True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')  # 周期时间
    iter_timer = Timer('m')  # 迭代时间
    best_psnr = -1

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        progress = tqdm(dataloader, desc=f"Epoch {epoch_idx}", leave=False)
        # Gradient accumulation setup
        accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))
        optimizer.zero_grad(set_to_none=True)
        for iter_idx, batch_data in enumerate(progress):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)  # 将数据加载到指定的设备上

            # Accumulated mixed precision forward
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_img = model(input_img)  # 将张量输入模型
                label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')  # 下采样
                label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')  # 下采样四倍
                l1 = criterion(pred_img[0], label_img4)  # 计算损失函数
                l2 = criterion(pred_img[1], label_img2)
                l3 = criterion(pred_img[2], label_img)
                loss_content = l1 + l2 + l3  # 损失函数

            # Replace deprecated torch.rfft with torch.fft.fftn and convert complex to real-imag view
            # Keep full spectrum (onesided=False equivalent) over last 2 dims (H, W), no normalization (default)
            def fft2_as_real(x: torch.Tensor) -> torch.Tensor:
                # x shape: [N, C, H, W]
                c = torch.fft.fftn(x, s=None, dim=(-2, -1), norm=None)
                # view as real-imag pairs, shape becomes [N, C, H, W, 2]
                return torch.view_as_real(c)

            label_fft1 = fft2_as_real(label_img4)
            pred_fft1 = fft2_as_real(pred_img[0])
            label_fft2 = fft2_as_real(label_img2)
            pred_fft2 = fft2_as_real(pred_img[1])
            label_fft3 = fft2_as_real(label_img)
            pred_fft3 = fft2_as_real(pred_img[2])

            f1 = criterion(pred_fft1, label_fft1)  # 经过傅里叶变换之后的Loss损失
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1 + f2 + f3

            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = loss_content + 0.1 * loss_fft  # 总的loss损失,原来是0.1倍的loss_fft
            # Normalize loss by accumulation steps
            loss = loss / accum_steps
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # Step when reaching accumulation boundary or last batch
            do_step = ((iter_idx + 1) % accum_steps == 0) or (iter_idx + 1 == max_iter)
            if do_step:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            iter_pixel_adder(loss_content.item())  # 每次迭代之后
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())  # 内容损失
            epoch_fft_adder(loss_fft.item())  # 快速傅里叶变换损失
            # print("'iter_idx + 1'=", iter_idx + 1)

            # Update tqdm status bar
            progress.set_postfix({
                'pix': f"{iter_pixel_adder.average():.4f}",
                'fft': f"{iter_fft_adder.average():.4f}"
            })

            if (iter_idx + 1) % args.print_freq == 0:  # 每100次迭代保存一次临时的model，显示一次
                lr = check_lr(optimizer)  # 检查lr
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(),
                                  iter_idx + (epoch_idx - 1) * max_iter)  # 计算像素损失，保存文件中供可视化使用
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()  # 每次迭代之后重置
                iter_fft_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch_idx}, overwrite_name)
        print(f"[Checkpoint] Overwrite latest: {overwrite_name}")

        if epoch_idx % args.save_freq == 0:  # save_freq=100，每100个周期保存一次模型
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
            print(f"[Checkpoint] Saved periodic: {save_name}")
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()  # 时间调度器计数
        if epoch_idx % args.valid_freq == 0:  # 每100个周期计算一次原始数据平均的PSNR
            val_gopro = _valid(model, args, epoch_idx)  # 计算一下原始数据集的平均峰值信噪比
            print("val_gopro==", val_gopro)  # 已修改
            print("epoch_idx==", epoch_idx)  # 已修改
            print('%03d epoch \n Average GOPRO PSNR %.2f dB' % (epoch_idx, val_gopro))  # 100个周期的平均峰值信噪比
            writer.add_scalar('PSNR_GOPRO', val_gopro, epoch_idx)  # 将所需的数据保存在文件里进行可视化，用来画图
            if val_gopro >= best_psnr:  # 平均的峰值信噪比大于-1,就可以保存模型
                best_path = os.path.join(args.model_save_dir, 'Best.pkl')
                torch.save({'model': model.state_dict()}, best_path)
                print(f"[Checkpoint] Saved best: {best_path}")
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
    print(f"[Checkpoint] Saved final: {save_name}")
