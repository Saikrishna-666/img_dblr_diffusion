import os
import torch
import argparse
from torch.backends import cudnn
from models.MRDNet import build_net, MRDNet
try:
    from models.full_model import FullModel
except ImportError:
    FullModel = None
from train import _train
from eval import _eval

def build_model(args):
    if args.use_dbrs:
        if FullModel is None:
            raise RuntimeError("FullModel not available; ensure models/full_model.py exists.")
        return FullModel(model_name=args.model_name,
                         use_dfd=args.use_dfd,
                         dbrs_depth=args.dbrs_depth,
                         dbrs_steps=args.dbrs_steps)
    if args.use_dfd and args.model_name == 'MRDNet':
        return MRDNet(use_dfd=True)
    return build_net(args.model_name)

def main(args):
    cudnn.benchmark = True
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    model = build_model(args)
    if torch.cuda.is_available():
        if args.mode == 'train' and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.cuda()
    if args.mode == 'train':
        _train(model, args)
    else:
        _eval(model, args)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', default='MRDNet', choices=['MRDNet','MRDNetPlus'])
    p.add_argument('--mode', default='train', choices=['train','test'])
    p.add_argument('--data_dir', type=str, default='/kaggle/input/go-pro/GOPRO')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--num_epoch', type=int, default=100)
    p.add_argument('--print_freq', type=int, default=10)
    p.add_argument('--num_worker', type=int, default=8)
    p.add_argument('--save_freq', type=int, default=10)
    p.add_argument('--valid_freq', type=int, default=10)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--start_epoch', type=int, default=1)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--lr_steps', type=list, default=[(x+1)*500 for x in range(3000//500)])
    p.add_argument('--train_proportion', type=float, default=1.0)
    p.add_argument('--crop_size', type=int, default=256)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--use_amp', type=bool, default=True)
    p.add_argument('--use_dfd', action='store_true')
    p.add_argument('--use_dbrs', action='store_true')
    p.add_argument('--dbrs_depth', type=int, default=48)
    p.add_argument('--dbrs_steps', type=int, default=4)
    p.add_argument('--refine_weight', type=float, default=0.5)
    p.add_argument('--freeze_mrd_epochs', type=int, default=0)
    p.add_argument('--mrd_lr_scale', type=float, default=0.2)
    p.add_argument('--freeze_scope', type=str, default='encoder', choices=['encoder','all','none'])
    p.add_argument('--checkpoint_dir', type=str, default='')
    p.add_argument('--test_model', type=str, default='/kaggle/input/mrdnet-checkpoint/Pretrained Model/MRDNet.pkl')
    p.add_argument('--save_image', type=bool, default=True, choices=[True, False])
    p.add_argument('--save_limit', type=int, default=0)
    args = p.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    if args.checkpoint_dir:
        args.model_save_dir = args.checkpoint_dir
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')
    print(args)
    main(args)
