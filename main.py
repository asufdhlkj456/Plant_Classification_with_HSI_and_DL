import os
import argparse
import tensorflow as tf

import ours_v5_1 as model     ##換版本記得改

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--PCs', type=int, default=6, help='how many PCA components, n= 1,2,3,...')
    parser.add_argument('--nbofdata', type=int, default=10, help='how many data of each class for training')        #20
    parser.add_argument('--crop_size', type=int, default=400, help='initial patch cropped from original cub')

    parser.add_argument('--input_size', type=int, default=(224,224), help='input size；resized of patch')      #80#100
    parser.add_argument('--num_bands', type=int, default=9, help='how many bands have choosed to train (Uniform)')
    parser.add_argument('--batch_size', type=int, default=12, help='training batch size')       #input size80 可以48  ##32
    parser.add_argument('--epoch', type=int, default=200, help='training epoch')     #先從程式裡設定  #500
    parser.add_argument('--lr', type=float, default=(1e-3), help='Initial learing rate')     #先從程式裡設定 1e-3

    parser.add_argument('--Mode', type=bool, default=True, help='Training or Testing... True-->Training ; False-->Testing')

    parser.add_argument('--test_step', type=int, default=20000, help='Testing step')         ## 32-->
    args = parser.parse_args()

    return args


def main(_):   ##python main.py
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    HSI = model.HSI(args)


    if args.Mode:
        HSI.train()     ###好像CPU 30% 左右 兩個load 的速度差不多GPU也都是10%左右

    else:
        HSI.test3()

if __name__ == '__main__':
    tf.app.run()

