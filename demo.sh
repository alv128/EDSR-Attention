# This file contains the commands to train and test the models.

# Training commands
# EDSR: Enhanced Deep Residual Network for Single Image Super-Resolution
python main.py --model EDSR --scale 4 --n_resblocks 8 --n_feats 64 \
--data_test Set5 --patch_size 192 --save edsr_x4 --save_models --save_results \
--epochs 1000 --decay 200-400-600-800 "$@"

# PA_EDSR: EDSR with Pyramid Attention
python main.py --model PA_EDSR --scale 4 --n_resblocks 8 --n_feats 64 --reduction 2 \
--data_test Set5 --patch_size 192 --save pa_edsr_x4 --save_models --save_results \
--epochs 1000 --decay 200-400-600-800 "$@"

# CA_EDSR: EDSR with Patch-based Correlation Attention
python main.py --model CA_EDSR --scale 4 --n_resblocks 8 --n_feats 64 \
--data_test Set5 --patch_size 192 --save ca_edsr_x4 --save_models --save_results \
--epochs 1000 --decay 200-400-600-800 "$@"

# ESA_EDSR: EDSR with Enhanced Spatial Attention
python main.py --model ESA_EDSR --scale 4 --n_resblocks 8 --n_feats 64 \
--data_test Set5 --patch_size 192 --save esa_edsr_x4 --save_models --save_results \
--epochs 1000 --decay 200-400-600-800 "$@"

# NLESA_EDSR: EDSR with Non-Local Enhanced Spatial Attention
python main.py --model NLESA_EDSR --scale 4 --n_resblocks 8 --n_feats 64 \
--data_test Set5 --patch_size 192 --save nlesa_edsr_x4 --save_models --save_results \
--epochs 1000 --decay 200-400-600-800 "$@"

# ESCA_EDSR: EDSR with Spatial Attention via Efficient Channel Attention
python main.py --model ESCA_EDSR --scale 4 --n_resblocks 8 --n_feats 64 \
--data_test Set5 --patch_size 192 --save esca_edsr_x4 --save_models --save_results \
--epochs 1000 --decay 200-400-600-800 "$@"


# Testing. Here, the trained models used in the report are loaded, which can be found in the folder
# 'pretrained_models'. These will give the results from the report.
# EDSR
python test.py --model EDSR --scale 4 --n_resblocks 8 --n_feats 64 \
--cpu --lpips --pre_train pretrained_models/EDSR_X4.pt --save_test_results "$@"

# PA_EDSR
python test.py --model PA_EDSR --scale 4 --n_resblocks 8 --n_feats 64 --reduction 2 \
--cpu --lpips --pre_train pretrained_models/PA_EDSR_X4.pt --save_test_results "$@"

# CA_EDSR
python test.py --model CA_EDSR --scale 4 --n_resblocks 8 --n_feats 64 \
--cpu --lpips --pre_train pretrained_models/CA_EDSR_X4.pt --save_test_results "$@"

# ESA_EDSR
python test.py --model ESA_EDSR --scale 4 --n_resblocks 8 --n_feats 64 \
--cpu --lpips --pre_train pretrained_models/ESA_EDSR_X4.pt --save_test_results "$@"

# NLESA_EDSR
python test.py --model NLESA_EDSR --scale 4 --n_resblocks 8 --n_feats 64 \
--cpu --lpips --pre_train pretrained_models/NLESA_EDSR_X4.pt --save_test_results "$@"

# ESCA_EDSR
python test.py --model ESCA_EDSR --scale 4 --n_resblocks 8 --n_feats 64 \
--cpu --lpips --pre_train pretrained_models/ESCA_EDSR_X4.pt --save_test_results "$@"
