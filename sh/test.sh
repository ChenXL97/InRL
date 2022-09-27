cd ../..
#python train.py task=DogMorph test=True num_envs=1 headless=False make_video=True checkpoint='runs/DogMorph/nn/DogMorph.pth' task.video.file_name='dog_morph.mp4'
python train.py \
task=Dog \
train.params.algo.name=a2c_continuous \
test=True \
num_envs=1024 \
headless=True \
make_video=False \
checkpoint='outputs/Dog/nn/Dog.pth' \
task.video.file_name='dog_mix.mp4' \
