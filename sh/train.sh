cd ..

train_name=$1
exp_name=$2
num_heads=$3
batch_morph=$4
total_morph=$5
offest=$6


for((i=($total_morph)*($offest) ;i<($total_morph)*($offest+1); i=( i+$batch_morph )))
do
  end=$(($i+$batch_morph))
  epm=$((8192 / $batch_morph))
  gtt=$(($epm * 8))

  python train.py \
  train=$train_name \
  experiment=$exp_name \
  num_heads=$num_heads \
  num_morphs=$batch_morph \
  train_envs_per_morph=$epm \
  test_envs_per_morph=$epm \
  games_to_track=$gtt \
  morph_asset_range="[$i,$end]"
done
