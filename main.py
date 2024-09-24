import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import os
import time

# 设置可见的GPU设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import argparse

from torch import nn
from torch.backends import cudnn
from models.DDLNet import build_net

from eval import _eval
from train import _train


def main(args):
    cudnn.benchmark = True
    mode = [args.mode, args.test_data]
    model = build_net(mode)
    model.cuda()
    pytorch_total_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print("\nTotal_params: ==> {}\n".format(pytorch_total_params / 1e6))
    # print(model)
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count():
        print("\nLet's use", torch.cuda.device_count(), "GPUs!\n")

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    if args.mode == 'train':
        _train(model, args)
    elif args.mode == 'test':
        _eval(model, args)


def setup_seed(seed=3407):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 选择确定性算法
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False


# 设置随机数种子
setup_seed(3407)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='DDLNet', type=str)
    parser.add_argument('--data_dir', type=str, default='/home/wuyabo/datasets/Dense_Haze')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--train_data', type=str, default='ITS-train')
    parser.add_argument('--test_data', type=str, default='ITS-test')
    parser.add_argument('--valid_data', type=str, default='ITS-test')
    parser.add_argument('--ex', type=str, default=f"{time.strftime('%m%d_%H-%M', time.localtime())}")

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--debug', type=bool, default=False, choices=[True, False])

    # Test
    parser.add_argument('--test_model', type=str, default='./SOTS-Indoor.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, args.train_data,
                                       f"weight_{args.ex}/")

    args.log_save_dir = os.path.join('results/', args.model_name, args.train_data,
                                     f"log_{args.ex}/")

    args.result_dir = os.path.join('results/', args.model_name, args.train_data,
                                   f"images_{args.ex}", args.train_data)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    if not os.path.exists(args.log_save_dir):
        os.makedirs(args.log_save_dir)

    command = 'cp ' + 'models/*.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + '*.py ' + args.model_save_dir
    os.system(command)
    
    print(f"\n\ntrain_{args.model_name}_{args.train_data}_{time.strftime('%Y%m%d_%H:%M', time.localtime())}\n\n")
    print(args)
    # print(torch.cuda.device_count())

    main(args)
