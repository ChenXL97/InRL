cd ../..

declare -a arg_list=($@)


for((i=$1 ;i<$(($1+$2)); i=i+1))
do
  start=$(($i*32))
  end=$(($i*32+32))

  python train.py \
  morph=Fix \
  train=MorphTensor \
  num_morphs=32 \
  train_envs_per_morph=256 \
  max_iterations=500 \
  test_envs_per_morph=64 \
  games_to_track=128 \
  morph_range=True \
  morph_asset_range="[$start,$end]" \
  ${arg_list[@]:2}


done
