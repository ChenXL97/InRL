cd ../..

python train.py \
morph=Fix \
train=SinglePPO \
num_morphs=1 \
morph_range=False \
train_envs_per_morph=4096 \
test_envs_per_morph=64 \
games_to_track=128 $@
