NTU Final Year Project (FYP): Facial Expression Transfer using StyleGAN
-----------------------------------------------------------------------

<p align="left"><img src="https://github.com/nancycyzl/NTU_FYP_Facial_Expression_Transfer_StyleGAN/blob/master/demo_image.PNG" width="90%"></p>

This project supports image and video-based facial expression transfer. <br>
Accompanying video [here](https://youtu.be/cIhcJgyeGB8)


## *Prerequisite*

### 1. Environment

- python 3.8.8
- torch 1.9.0+cu111
- torchvision 0.10.0+cu111
- cuda 11.7
- pillow 9.3.0
- numpy 1.10.1
- ninja 1.10.2.3
- requests 2.25.1
- pathlib2 2.3.5
- click 7.1.2
- tqdm 4.35.0



- Visual Studio 2019 (for windows to compile models)


### 2. Resources to be downloaded

1. data_processing/resources: download "shape_predictor_68_face_landmarks.dat" from [here](https://drive.google.com/file/d/1i33vr1M_VqIbD4_2CUQNff82t6GFI4zz/view?usp=share_link)
2. pSp/pretrained: download "CurricularFace_Backbone.pth" from [here](https://drive.google.com/file/d/10z6DrOecDlCngAPfbaCKD2XF1faHtiHN/view?usp=sharing)
3. pSp/pretrained: download "psp_ffhq_encode.pt" from [here](https://drive.google.com/file/d/1kLvdEaiZBDOzKUTujutjZ209VopCZZp_/view?usp=sharing)
4. styleGAN2/pretrained: download "metfaces.pkl" from [here](https://drive.google.com/file/d/1fV-mdaar4ETbXWPdxOqBHh4xML1ntyNH/view?usp=sharing)
5. styleGAN2/pretrained: download "stylegan2-ffhq-config-f.pkl" from [here](https://drive.google.com/file/d/1eFmb7Czqp3Po1oMCA682MWGGXoaegOpW/view?usp=sharing)
6. styleGAN2/resources: download "ffhq-dataset-v2.json" from [here](https://drive.google.com/file/d/1dVLoUL1xw7SX5q-tDjajaAZLanciA9RN/view?usp=sharing)


## *Code usage*

### 1. Data Processing

Change directory to "data_processing"

#### a) Video processing

Mode 1: frames to video<br>
Mode 2: video to frames<br><br>
NOTE: rmb to change file address in the script

```
python video_frames.py
```

#### b) Face alignment

Crop and align the face. Input could be an image or a folder of images.<br>
Enable_smooth is only applicable to processing a folder of images. It assumes the person's head to stay in the same position,
so the coordinates/angle of the face is only computed using the first image and applied to all the rest of images.

```
python face_alignment.py input_path output_path --enable_smooth
```

#### c) Latent code analysis

latent_code_analyzing.ipynb file: calculate and plot the magnitude of 18 layers of *code_exp - code_neu*

### 2. Generate images using StyleGAN2

Change directory to "styleGAN2"

#### a) generate random images using StyleGAN2

```
python generate.py --outdir=out --seeds=200-205 --network=pretrained/stylegan2-ffhq-config-f.pkl
```

#### b) Project an image to the latent space of StyleGAN2 (W space)

Obtain the latent code of an arbitrary image (in w space) using optimization method (default 1000 steps).<br>
Results include: 1) projeted image; 2) w latent code; 3) optimization progress video (set save-video to False to disable)

```
python projector.py --outdir=out --target=..data/frame18.jpg --network=pretrained/stylegan2-ffhq-config-f.pkl (--save-video False --num-steps 1000)
```

### 3. Generate images using pSp
Change directory to pSp

#### a) Embed arbitrary images to the latent space of StyleGAN2 (W+ space) using pSp encoder

parameters:<br>
data_path: a folder containing image(s) to be embedded <br>
exp_dir: result saving path <br><br>
NOTE: if want to save inferenced latent code, go to models/stylegan2/model.py & uncomment line 489-503
& change w_plus_folder accordingly (recommend: exp_dir/w_plus)
```
python scripts/inference.py --data_path data/images --checkpoint_path pretrained/psp_ffhq_encode.pt --test_batch_size=1 --exp_dir data/results
```
Alternatively, just type```.\commands.bat``` in terminal

#### b) Manipulate w+ latent code

parameters:<br>
--target: latent code file (.npy) of target face <br>
--source_exp: latent code file (.npy) of target expression <br>
--source_neu: latent code file (.npy) of neutral expression <br>
--alpha: alpha value that controlling transfer intensity <br>
--save_dir: saving directory
```
python explore_w_plus.py --target target_code_file --source_exp source_exp_code_file --source_neu source_neu_code_exp ^
       --alpha alpha_value  --save_dir saving_directory
```

#### c) Generate image from w+ latent code

parameters:<br>
input_path: a file or a folder of files containing w+ latent code (.npy) <br>
output_path: could be an image filename (jpg/png) or a folder(default jpg)<br><br>
NOTE: go to models/stylegan2/model.py & comment out line 489-503
```
python image_from_w_plus.py input_path output_path
```

#### d) Transfer expression for videos

This script takes target face (npy file) and driving video (a folder of npy files for all frames) as input, and outputs
the transferred frames of the target video. Before running this scripts, users should have prepared all the npy files
by running inference.py. Users may use data/processing/video_frames.py to combine all the frames to get the transferred video.

parameters: <br>
target: the w+ latent code (.npy) for the target face <br>
driving_source: a folder of latent code files where each file is for one frame of the driving video<br>
source_neutral: a manually selected neutral expression (.npy) from the driving video<br>
--seq_file: the sequence file generated during inference, which is used for combining transferred frames in correct sequence<br>
--save_dir: result saving directory<br>
--alpha: a float value controlling transfer intensity<br>
--save_transferred_code: whether save the intermediate transferred code, which is used to generate transferred frames<br><br>
NOTE: go to models/stylegan2/model.py & comment out line 489-503<br>
```
python driving_video_pSp_byFrames.py target driving_source source_neutral --seq_file --save_dir --alpha --save_transferred_code
```

## *Acknowledgement*

Some codes are adapted from the following repository: <br>
https://github.com/NVlabs/stylegan2-ada-pytorch <br>
https://github.com/eladrich/pixel2style2pixel <br>