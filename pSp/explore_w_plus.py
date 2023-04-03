'''
This script takes two latent codes as input and output the resulting latent code based on alpha
code_result = code_target + alpha * (code_expression - code_neutral)

usage: python explore_w_plus.py --target target_code_file --source_exp source_exp_code_file --source_neu source_neu_code_exp ^
       --alpha alpha_value  --save_dir saving_directory
'''
import numpy as np
import argparse
import torch
import os

def parse_args():
    parse = argparse.ArgumentParser(description='generate images from w+ latent codes')
    parse.add_argument('--target', type=str, help='latent code of target face (.npy)')
    parse.add_argument('--source_exp', type=str, help='latent code of source expression')
    parse.add_argument('--source_neu', type=str, help='latent code of neutral expression of the source identity')
    parse.add_argument('--alpha', type=float, default=0.6, help='alpha value controlling transfer intensity')
    parse.add_argument('--save_dir', type=str, help='results saving folder')
    args = parse.parse_args()
    return args

def check_paths(tgt, output, alpha):   # args.save_dir
    # check input files
    if not (tgt.endswith('.npy') and tgt.endswith('.npy') and tgt.endswith('.npy')):
        print('Input latent codes must be .npy files')

    # output dir
    if output:
        if not os.path.exists(output):
            os.makedirs(output)
    else:
        # user has not specified save directory, use default one
        folder, filename = os.path.split(tgt)
        filename = filename.split('.')[0]
        output = os.path.join(folder, filename, f'alpha_{alpha}')
        os.makedirs(output)

    return output


def main(args):

    output = check_paths(args.target, args.save_dir, args.alpha)

    # latent minus
    code_base = np.load(args.target)
    code_exp = np.load(args.source_exp)
    code_neu = np.load(args.source_neu)

    # method 1: direct subtraction
    code_new = code_base + args.alpha*(code_exp-code_neu)

    # method 2: layer 3 and 5
    # this only copy the reference, so the change in code_new will also change code_base

    base_layer3 = code_base[0][2] + args.alpha * (code_exp - code_neu)[0][2]
    base_layer5 = code_base[0][4] + args.alpha * (code_exp - code_neu)[0][4]
    code_new = code_base
    code_new[0][2] = base_layer3
    code_new[0][4] = base_layer5


    filename = f'w_plus_{args.alpha}.npy'
    np.save(os.path.join(output, filename), code_new)
    print(f'Done! Result saved in {output}')


if __name__ == '__main__':
    args = parse_args()
    main(args)

