import argparse
from solver import Solver

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_root', type=str, default='/BS/databases/CelebA')
    parser.add_argument('--gamma', type=float, default=6.4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_iter', type=int, default=150000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--cp_dim', type=int, default=4)
    parser.add_argument('--cp_ext', type=int, default=3)
    parser.add_argument('--cu_dim', type=int, default=40)
    parser.add_argument('--img_dir', type=str, default='./imgs')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()
    print (args.eval)
    solver = Solver(args)
    if args.eval:
        print ('eval mode')
        solver.eval()
    else:
        solver.train()

if __name__ == "__main__":
    main()
