'''
This script takes target face (npy file) and driving video (a folder of npy files for all frames) as input, and outputs
the transferred frames of the target video. Before running this scripts, users should have prepared all the npy files
by running inference.py. Users also need to manually choose a neutral expression from all the frames of the driving video
and specify it via the source_neutral variable. In models/stylegan2/model.py, users need to comment out line 489-503.

Usage: python driving_video_pSp_byFrames.py target driving_source source_neutral --seq_file --save_dir --alpha --save_transferred_code

target: the w+ latent code (.npy) for the target face - the face to change expressions
driving_source: a folder of latent code files where each file is for one frame of the driving video
source_neutral: a latent code (.npy) for the neutral expression from the driving video
--seq_file: the sequence file generated during inference, which is used for combining transferred frames in correct sequence
--save_dir: result saving directory
--alpha: a float value controlling transfer intensity
--save_transferred_code: if don't want save then don't include this argument
'''

from models.stylegan2.model import Generator
from PIL import Image
import numpy as np
import argparse
import torch
import os
import re


def parse_args():
    parse = argparse.ArgumentParser(description='transfer facial expression for videos')
    parse.add_argument('target', type=str, help='w+ latent code file (.npy) for the target image')
    parse.add_argument('driving_source', type=str,
                       help='a folder of latent code files, each file for each frame of the driving video')
    parse.add_argument('source_neutral', type=str,
                       help='a w+ latent code (.npy) for the neutral expression from the driving video')
    parse.add_argument('--seq_file', type=str, help='the sequence file generated during inference')
    parse.add_argument('--save_dir', type=str, help='result saving directory')
    parse.add_argument('--alpha', type=float, default=0.6, help='alpha value for transfer intensity')
    parse.add_argument('--save_transferred_code', action='store_true',
                       help='whether to save intermediate transferred latent code (for generating transferred frames)')
    args = parse.parse_args()
    return args


def transfer(code_base, code_neu, code_exp, alpha=0.6):
    return code_base + alpha * (code_exp - code_neu)


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


if __name__ == '__main__':
    args = parse_args()
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    code_base = np.load(args.target)
    code_neu = np.load(args.source_neutral)

    print('Loading weights...')
    ckpt = torch.load('pretrained/psp_ffhq_encode.pt', map_location='cpu')
    generator = Generator(1024, 512, 8).cuda()
    generator.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)

    if args.save_transferred_code:
        save_dir_np = os.path.join(save_dir, 'transferred_code')
        os.makedirs(save_dir_np)

    save_dir_frames = os.path.join(save_dir, 'transferred_frames')
    os.makedirs(save_dir_frames, exist_ok=True)

    with open(args.seq_file, 'r') as f:
        lines = f.readlines()
        for line in lines:  # loop through all frames by seq number
            seq_count, img_path = line.strip('\n').split(' ')  # str
            frame_count = re.findall('\d+', os.path.basename(img_path))[0]  # str

            exp_code_file = os.path.join(args.driving_source, 'seq_' + seq_count + '.npy')
            code_exp = np.load(exp_code_file)

            result_code = transfer(code_base, code_neu, code_exp, alpha=args.alpha)

            if args.save_transferred_code:
                np_save_path = os.path.join(save_dir, 'transferred_code', 'frame' + frame_count + '.npy')
                np.save(np_save_path, result_code)

            input = torch.tensor(np.array([result_code])).cuda()

            print('Generating image for seq count {}, frame {}'.format(seq_count, frame_count))
            images, result_latent = generator(input, input_is_latent=True, randomize_noise=False, return_latents=False)

            result = tensor2im(images[0])
            im_save_path = os.path.join(save_dir_frames, 'frame{}.jpg'.format(frame_count))
            Image.fromarray(np.array(result)).save(im_save_path)
        print(f'Results saved at {save_dir}')
