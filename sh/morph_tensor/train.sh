cd ../..

batch_morph=$2
total_morph=$3
offest=$4

for((i=($total_morph)*($offest) ;i<($total_morph)*($offest+1); i=( i+$batch_morph )))
do
  end=$(($i+$batch_morph))
  epm=$((8192 / $batch_morph))

  python train.py \
  experiment=$1 \
  train=MorphTensor \
  num_morphs=$batch_morph \
  train_envs_per_morph=$epm \
  max_iterations=500 \
  morph_asset_range="[$i,$end]"

done
