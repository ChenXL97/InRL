cd ../..

batch_morph=$2
total_morph=$3
offest=$4

for((i=($total_morph)*($offest) ;i<($total_morph)*($offest+1); i=( i+$batch_morph )))
do
  end=$(($i+$batch_morph))
  epm=$((8192 / $batch_morph))
  gtt=$(($epm * 8))

  python train.py \
  experiment=$1 \
  test=True \
  save_morph=True \
  train=MorphTensor \
  num_morphs=$batch_morph \
  test_envs_per_morph=$epm \
  games_to_track=$gtt \
  max_iterations=500 \
  morph_asset_range="[$i,$end]"

done
