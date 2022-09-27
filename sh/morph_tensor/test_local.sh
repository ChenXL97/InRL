cd ../..

declare -a arg_list=($@)


python test.py -cp outputs/${arg_list[0]}/cfg \
test=True \
+morph=Fix \
+train=MorphTensor \
num_morphs=1 \
test_envs_per_morph=1 \
games_to_track=128 \
morph_range=True \
headless=False \
${arg_list[@]:1}

