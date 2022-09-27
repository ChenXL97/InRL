cd ../..

declare -a arg_list=($@)

python test.py -cp ~/outputs_server/${arg_list[0]}/cfg \
test=True \
headless=False \
make_video=True \
games_to_track=10 \
num_morphs=1 \
morph_range=False \
train_envs_per_morph=4096 \
test_envs_per_morph=1 \
${arg_list[@]:1}