# -*- coding: utf-8 -*-
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--image_scale', type=int, default=64, help='Image scale default 64')
parser.add_argument('--noise_scale', type=int, default=64, help='Noise scale default 64')
parser.add_argument('--batch_size', type=int, default=16,  help='Number of batch size')
parser.add_argument('--hidden_num', type=int, default=128, help='Hidden number in G&D network')

parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lambda_k', type=float, default=0.001)
parser.add_argument('--d_lr', type=float, default=0.00008)
parser.add_argument('--g_lr', type=float, default=0.00008)
parser.add_argument('--i_lr', type=float, default=0.001)
parser.add_argument('--d_lr_low_boundary', type=float, default=0.00002)
parser.add_argument('--g_lr_low_boundary', type=float, default=0.00002)
parser.add_argument('--random_seed', type=int, default=0)

parser.add_argument('--max_step', type=int, default=300000)
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--save_step', type=int, default=5000)
parser.add_argument('--lr_update_step', type=int, default=100000)
parser.add_argument('--interp_step', type=int, default=2000)

parser.add_argument('--mode', type=int, default=0, choices=[0,1,2,3], 
                        help='Run mode, 0: network trainint, \
                                        1: single image horizental flip interpolate,\
                                        2: two images interpolate,\
                                        3: auto interpolation(image from dataset)')
parser.add_argument('--image1', type=str, help='The first interpolate image name')
parser.add_argument('--image2', type=str, help='The second interpolate image name')
parser.add_argument('--out_dir', type=str, default='data/output', help='Output path')
parser.add_argument('--interp_dir', type=str, default='data/interpolate', help='Interpolate path')
parser.add_argument('--data_dir', type=str, default='/home/zyl8129/Documents/datasets/img_align_celeba', help='Dataset path')
args = parser.parse_args()


if __name__ == '__main__':
    # set numpy seed
    np.random.RandomState(seed=args.random_seed)
	
    if args.mode == 0:
        from tools.train import Trainer
        t = Trainer(args)
        t.train()
    else:
        from tools.sample import Sampler
        s = Sampler(args)
        s.interpolate()
