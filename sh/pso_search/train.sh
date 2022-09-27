cd ../..


python train.py \
morph=PSOSearch \
morph.asset.from_file=False \
train=MorphTensor \
num_morphs=32 \
train_envs_per_morph=256 \
test_envs_per_morph=64 \
games_to_track=128 \
max_search_iters=50 $@
