@echo on

:: project an image (output: i024x1024 face image / latent code npz / transition video)
:: pretrained/stylegan2-ffhq-config-f.pkl
:: ../temp_from_colab/weight1.pkl
python projector.py --network pretrained/stylegan2-ffhq-config-f.pkl --target ../data/images/img19_aligned.jpg ^
 --save-video True --outdir ../data/projection_results/result19 --seed 408 --num-steps 700

:: train, pretrained FFHQ as base
::python train.py --outdir training_runs --data data/train_set_test --gpus 1 --resume pretrained/stylegan2-ffhq-config-f.pkl ^
:: --snap 10 --batch 2

::  style mixing
:: python style_mixing.py --outdir ../data/style_mixing --rows=85,100,75 --cols=55,821,1789 ^
    ::--network pretrained/stylegan2-ffhq-config-f.pkl