cd ../..

declare -a arg_list=($@)


python train.py \
morph=Fix \
task=$1 \
train=MorphTensor \
num_morphs=32 \
train_envs_per_morph=256 \
max_iterations=500 \
test_envs_per_morph=64 \
games_to_track=128 \
morph_range=True \
morph_asset_range="[0,32]" \
count_time=True \
${arg_list[@]:1}

