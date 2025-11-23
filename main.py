import os
import torch
import argparse
from torch.backends import cudnn
from models.MRDNet import build_net, MRDNet
from train import _train
from eval import _eval


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Instantiate model; allow optional DFD activation via command-line flag (MRDNet only)
    if getattr(args, 'use_dfd', False) and args.model_name == 'MRDNet':
        model = MRDNet(use_dfd=True)
    else:
        model = build_net(args.model_name)
    # Multi-GPU support: only wrap with DataParallel during training.
    if torch.cuda.is_available():
        if args.mode == 'train' and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)
        model = model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='MRDNet', choices=['MRDNet', 'MRDNetPlus'], type=str)
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/go-pro/GOPRO')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to start from when resuming from weights-only file')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])
    parser.add_argument('--train_proportion', type=float, default=1.0, help='Proportion of training data to use (0-1]')
    parser.add_argument('--use_dfd', action='store_true', help='Enable Dynamic Frequency Decomposition (DFD) modules')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='Directory to save checkpoints; overrides default results/<model_name>/weights')
    parser.add_argument('--crop_size', type=int, default=256, help='Training crop size. Set 0 to disable cropping and use full images')
    parser.add_argument('--use_amp', type=bool, default=True, help='Use mixed precision (AMP) to save memory')

    # Test
    parser.add_argument('--test_model', type=str, default='/kaggle/input/mrdnet-checkpoint/Pretrained Model/MRDNet.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
    parser.add_argument('--save_limit', type=int, default=0, help='If > 0, save at most this many images during evaluation')

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    if args.checkpoint_dir:
        args.model_save_dir = args.checkpoint_dir
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')

    # Ensure directories exist
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir, exist_ok=True)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)
    print(args)
    main(args)
