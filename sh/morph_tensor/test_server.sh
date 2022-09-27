cd ../..

declare -a arg_list=($@)


python test.py -cp ~/outputs_server/${arg_list[0]}/cfg \
+morph=Fix \
+train=MorphTensor \
+debug=False \
test=True \
num_morphs=1 \
test_envs_per_morph=64 \
games_to_track=128 \
morph_range=True \
${arg_list[@]:1}

