import warnings
warnings.filterwarnings("ignore")

import os
import time
import sys

# 设置可见的GPU设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import argparse

from torch import nn
from torch.backends import cudnn
#from models.MIMOUNet_sf_dcm_af_kernel5_v7 import build_net
from models.MIMOUNet_base_v5_22_1 import build_net
#from models.baseline4_1block import build_net
#from models.baseline_3res import build_net

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
    # elif args.mode == 'train' and args.data == 'Outdoor':
    #     _train_ots(model, args)
    elif args.mode == 'test':
        _eval(model, args)



class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='MIMOUNet', type=str)
    parser.add_argument('--data_dir', type=str, default='/home/wuyabo/datasets/Dense_Haze')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--train_data', type=str, default='train')
    parser.add_argument('--test_data', type=str, default='test')
    parser.add_argument('--valid_data', type=str, default='test')
    parser.add_argument('--ex', type=str, default=f"{time.strftime('%m%d_%H-%M', time.localtime())}")

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--debug', type=bool, default=False, choices=[True, False])

    # Test
    parser.add_argument('--test_model', type=str, default='/root/autodl-tmp/SFNet/SOTS-Indoor.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, args.train_data,
                                       f"weight_{args.ex}/")

    args.log_save_dir = os.path.join('results/', args.model_name, args.train_data,
                                     f"log_{args.ex}/")

    args.result_dir = os.path.join('results/', args.model_name, args.train_data,
                                   f"images_{args.ex}/")

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    if not os.path.exists(args.log_save_dir):
        os.makedirs(args.log_save_dir)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    command = 'cp ' + 'models/*.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + '*.py ' + args.model_save_dir
    os.system(command)
    # if args.data == 'Indoor':
    #     command = 'cp ' + 'train.py ' + args.model_save_dir
    #     os.system(command)
    # elif args.data == 'Outdoor':
    #     command = 'cp ' + 'train_ots.py ' + args.model_save_dir
    #     os.system(command)

    # log_file = os.path.join(args.model_save_dir,
    #                         f"train_{args.model_name}_{time.strftime('%Y%m%d_%H-%M', time.localtime())}.log")
    # logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    #
    # args.logger = logger
    # args.logger.info(args)

    filename = args.log_save_dir + args.ex + ".log"
    sys.stdout = Logger(filename=filename, stream=sys.stdout)

    print(f"\n\ntrain_{args.model_name}_{args.train_data}_{time.strftime('%Y%m%d_%H:%M', time.localtime())}\n\n")
    print(args)
    # print(torch.cuda.device_count())

    main(args)
