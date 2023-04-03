'''
This script generates images from w+ latent code. In models/stylegan2/model.py, comment out line 489-503.

Usage: python image_from_w_plus.py input_path output_path

input_path: a file or a folder of files containing w+ latent code (.npy)
output_path: could be an image filename (jpg/png) or a folder(default jpg)
'''


from models.stylegan2.model import Generator
from PIL import Image
import numpy as np
import argparse
import torch
import tqdm
import glob
import sys
import os

sys.path.append('.')


def parse_args():
    parse = argparse.ArgumentParser(description='generate images from w+ latent codes')
    parse.add_argument('input_path', type=str, help='input: a w+ latent code file (.npy) or a folder of latent code')
    parse.add_argument('output_path', type=str, help='output: an image filename or a new folder')
    args = parse.parse_args()
    return args


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

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def check_paths(input, output):
    is_file = 0
    if not os.path.exists(input):
        print(f'input path {input} does not exist!!')
        sys.exit()
    elif os.path.isfile(input):   # input is file
        is_file = 1
        if not output.endswith(('.jpg', '.png')):   # output is a folder name
            create_folder(output)
            _, tail = os.path.split(input)
            output = os.path.join(output, tail.replace('.npy', '.jpg'))
            print(output)

    elif os.path.exists(output):    # input is folder and output folder already exists
        print(f'Output folder {output} already exists!')
        sys.exit()
    else:
        os.makedirs(output)

    return input, output, is_file


def generate_one_image(latent_code, result_file_name):
    input = torch.tensor(np.array([latent_code])).cuda()
    images, result_latent = generator(input, input_is_latent=True, randomize_noise=False, return_latents=False)
    result = tensor2im(images[0])
    Image.fromarray(np.array(result)).save(result_file_name)


if __name__ == '__main__':
    args = parse_args()

    input_path, output_path, is_file = check_paths(args.input_path, args.output_path)

    print('Loading weights...')
    ckpt = torch.load('pretrained/psp_ffhq_encode.pt', map_location='cpu')
    generator = Generator(1024, 512, 8).cuda()
    generator.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)

    if is_file:
        print('Loading w plus latent code...')
        code_load = np.load(input_path)
        print('Generating...')
        generate_one_image(code_load, output_path)
    else:
        for code_file in tqdm.tqdm(glob.glob(input_path+'\*.npy')):
            code_load = np.load(code_file)
            _, tail = os.path.split(code_file)
            result_file_name = os.path.join(output_path, tail.replace('.npy', '.jpg'))
            generate_one_image(code_load, result_file_name)

    print(f'Result saved in {output_path}')
