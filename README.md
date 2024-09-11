# <img src="./assets/img1_title3.png" width="80" height="auto">threestudio-dreambeast
Runjia Li, Junlin Han, Luke Melas-Kyriazi, Chunyi Sun, Zhaochong An, Zhongrui Gui, Shuyang Sun, Philip Torr, Tomas Jakab

<a href='https://runjiali-rl.github.io/projects/dreambeast.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/xxx'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

The DreamBeast extension for <a href='https://github.com/threestudio-project/threestudio'>threestudio</a>. To use it, simply install this extension in threestudio `custom` directory.

<center><img src="./assets/Chimera teaser.png" alt="mainimg" style="width:850px"><center>

# Installation
```bash
cd custom
git clone https://github.com/runjiali-rl/threestudio-dreambeast.git
cd threestudio-dreambeast

# First install xformers (https://github.com/facebookresearch/xformers#installing-xformers)
# cuda 11.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
# cuda 12.1 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements.txt
```

# Quick Start
```bash
# Run the following commands in the threestudio repository

# Replace the OPENAI_API_KEY with your openai api key
python launch.py --config custom/threestudio-dreambeast/configs/dreambeast.yaml --train --gpu 0 system.prompt_processor.prompt="a creature with a body of a kangaroo and the shell of a tortoise" "system.api_key=OPENAI_API_KEY",

python launch.py --config custom/threestudio-dreambeast/configs/dreambeast.yaml --train --gpu 0 system.prompt_processor.prompt="a car with airplane wings" "system.api_key=OPENAI_API_KEY",

python launch.py --config custom/threestudio-dreambeast/configs/dreambeast.yaml --train --gpu 0 system.prompt_processor.prompt="An object with the screen of a television and the wings of a butterfly" "system.api_key=OPENAI_API_KEY",
```

# Citing

If you find DreamBeast helpful, please consider citing:

```
@article{li2024DreamBeast,
  author = {Runjia Li, Junlin Han, Luke Melas-Kyriazi, Chunyi Sun, Zhaochong An, Zhongrui Gui, Shuyang Sun, Philip Torr, Tomas Jakab},
  title = {DreamBeast: Distilling 3D Fantastical Animals with Part-Aware Knowledge Transfer},
  journal = {arXiv:2308.16512},
  year = {2023},
}
```
