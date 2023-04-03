'''
Usage: python scripts/inference.py --data_path data/images --checkpoint_path pretrained/psp_ffhq_encode.pt --test_batch_size=1  ^
  --exp_dir data/results
'''

import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp


def run():
    # print('inside the run() function')
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        w_plus_folder = os.path.join(test_opts.exp_dir, 'w_plus',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
        w_plus_folder = os.path.join(test_opts.exp_dir, 'w_plus')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)
    os.makedirs(w_plus_folder, exist_ok=True)

    # update test options with options used during training
    # print('line 40: loading checkpoint')
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    # print('line 42, loaded')
    opts = ckpt['opts']   # dict
    opts.update(vars(test_opts))  # test_opts is a Namespace; vars(test_opts) is a dict
    if 'learn_in_w' not in opts:  # the key does not exist
        opts['learn_in_w'] = False   # add the key to the dict
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)  # convert dict to Namespace again

    # creating pSp model
    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]  # a dict
    transforms_dict = dataset_args['transforms'](opts).get_transforms()   # a dict
    dataset = InferenceDataset(root=opts.data_path,   # a class inherited from torch Dataset
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,   # Dataloader also from torch
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    file_proc_seq = ''
    for input_batch in tqdm(dataloader):  # loop through all batches
        if global_i >= opts.n_images:
            break
        with torch.no_grad():   # disabling gradient calculation during inference (reduce memory consumption)
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch = run_on_batch(input_cuda, net, opts)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):  # loop through all images in one batch
            result = tensor2im(result_batch[i])
            im_path = dataset.paths[global_i]
            file_proc_seq += str(global_i)+' '+im_path+'\n'

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                if opts.resize_factors is not None:
                    # for super resolution, save the original, down-sampled, and output
                    source = Image.open(im_path)
                    res = np.concatenate([np.array(source.resize(resize_amount)),
                                          np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                          np.array(result.resize(resize_amount))], axis=1)
                else:
                    # otherwise, save the original and output
                    res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                          np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)
    with open(stats_path, 'w') as f:
        f.write(result_str)

    file_proc_seq_path = os.path.join(opts.exp_dir, 'file_proc_seq.txt')
    with open(file_proc_seq_path, 'w') as f:
        f.write(file_proc_seq)


def run_on_batch(inputs, net, opts):
    # inputs is one batch
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # print('In inference line 124 latent_to_inject: ', latent_to_inject.shape)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == '__main__':
    print('starting the program')
    run()
