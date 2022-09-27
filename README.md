# Instinct-driven Reinforcement Learning

Instinct-driven Reinforcement Learning for Embodied Intelligence


## Installation

1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
   - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)

4. Install RL-Games
   * https://github.com/Denys88/rl_games



## Usage


Run a simple training process by the following command (set `headless=True` to disable the viewer):

`python train.py task=Dog morph=PSOSearch train=MorphTensor headless=False`

## Settings

We use Hydra to manage configurations
 * https://hydra.cc/docs/intro/

Following main options are currently supported:

```
task: Kangaroo | Raptor | Dog | 
terrain: Flat | Uphill | Downhill | 
train: SinglePPO | MorphTensor 
morph: Fix | PSOSearch
```

See more arguments in `inrl/cfg`.