:: inference (rmb go to model.py line 484 to change w_plus.npy save address)
python scripts/inference.py --data_path ../data/jayzhou --checkpoint_path pretrained/psp_ffhq_encode.pt --test_batch_size=1  ^
  --exp_dir ../data/jayzhou_results
  