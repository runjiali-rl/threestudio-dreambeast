# threestudio-dreambeast
![DreamBeast](https://github.com/DSaurus/threestudio-DreamBeast/assets/24589363/b21e2a80-7ea9-4add-890e-0395b91aa5af)

The DreamBeast extension for threestudio. To use it, simply install this extension in threestudio `custom` directory.

# Installation
```
cd custom
git clone https://github.com/DSaurus/threestudio-DreamBeast.git
cd threestudio-dreamBeast

# First install xformers (https://github.com/facebookresearch/xformers#installing-xformers)
# cuda 11.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
# cuda 12.1 version
# pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements.txt
```

# Quick Start
```
# DreamBeast without shading (memory efficient)
python launch.py --config custom/threestudio-DreamBeast/configs/DreamBeast-sd21.yaml --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse"

# DreamBeast with shading (used in paper)
python launch.py --config custom/threestudio-DreamBeast/configs/DreamBeast-sd21-shading.yaml --train --gpu 0 system.prompt_processor.prompt="an astronaut riding a horse"
```

# Citing

If you find DreamBeast helpful, please consider citing:

```
@article{shi2023DreamBeast,
  author = {Shi, Yichun and Wang, Peng and Ye, Jianglong and Mai, Long and Li, Kejie and Yang, Xiao},
  title = {DreamBeast: Multi-view Diffusion for 3D Generation},
  journal = {arXiv:2308.16512},
  year = {2023},
}
```
