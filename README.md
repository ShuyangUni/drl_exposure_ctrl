<!--
 * @Author: Shuyang Zhang
 * @Date: 2024-08-25 19:55:54
 * @LastEditors: ShuyangUni shuyang.zhang1995@gmail.com
 * @LastEditTime: 2024-08-25 23:30:05
 * @Description: 
 * 
 * Copyright (c) 2024 by Shuyang Zhang, All Rights Reserved. 
-->
# Efficient Camera Exposure Control for Visual Odometry via Deep Reinforcement Learning

This is an official repository of

**Efficient Camera Exposure Control for Visual Odometry via Deep Reinforcement Learning**, Shuyang Zhang, Jinhao He, Yilong Zhu, Jin Wu, and Jie Yuan.

This paper is currently under review at **IEEE Robotics and Automation Letters (RAL)**.

<p align="center">
  <img src="docs/fw_train.png" width = "100%" alt="Training process." title="overview" />
</p>
<p align="center">
  <strong> Training process. </strong>
</p>

<p align="center">
  <img src="docs/fw_infer.png" width = "100%" alt="Inference process." title="overview" />
</p>
<p align="center">
  <strong> Inference process. </strong>
</p>

## Motivations
We want to implement an exposure control method based on deep reinforcement learning (DRL), which
* enables **fast** and **convenient** offline training via a simulation environment;
* adds **high-level information** to make the agent **intelligent** for subsequent visual tasks, for this paper, visual odometry (VO).

## Highlights
* **A DRL-based camera exposure control solution**. The exposure control challenge is divided into two subtasks, enabling completely offline DRL operations without the necessity for online interactions.
* **An lightweight image simulator** based on imaging principles,significantly enhances the data efficiency and simplifies the complexity of DRL training.
* **A study on reward function design** with various levels of information. The trained agents are equipped with different intelligence, enabling them to deliver exceptional performance in challenging scenarios.
* **Sufficient experimental evaluation**, which demonstrates that our exposure control method improves the performance of VO tasks, and achieves faster response speed and reduced time consumption.

<p align="center">
  <img src="docs/cover_01.png" width = "70%" alt="cover_01" title="cover_01" />
</p>
<p align="center">
  <img src="docs/cover_02.png" width = "70%" alt="cover_02" title="cover_02" />
</p>
<p align="center">
  Our DRL-based method with feature-level rewards (DRL-feat) exhibits a high-level comprehension of lighting and motion. It predicts the impending over-exposure event and preemptively reduces the exposure. While this adjustment temporarily decreases the number of tracked feature points, it effectively prevents a more severe failure in subsequent frames.
</p>

## Run the code
### Setup
1. Download our dataset. 
   If you only want to run the agents with the [pretrained model](model), please download the [test dataset](https://hkustconnect-my.sharepoint.com/:u:/g/personal/szhangcy_connect_ust_hk/Ef3AfNOkCLZKqXRKPYEbxjcBiHXBtSmV3-2IAT7xRdon_w?e=1jODLF) only.
   If you want to train with our data, please download the [full datasets](https://hkustconnect-my.sharepoint.com/:u:/g/personal/szhangcy_connect_ust_hk/EYSZnuAgTCJNnYqZpWaN6HYBPNHvzwkbjdxA7rzVhfyuTA?e=YWgyWN).

2. Configure the environment.
   Our code is implemented in Python. You can use Conda and Pip to install all the required packages.

   ```
   # create conda environment
   conda create -n drl_expo_ctrl python=3.8
   conda activate drl_expo_ctrl

   # install requirement packages
   pip install opencv-python pyyaml tensorboard

   # install torch, recommended to follow the official website guidelines with CUDA version.
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   ```

3. Update configuration and parameters
   After unzipping the datasets, please change the root directory (**seqs_root**) of the dataset in [train.yaml](config/train.yaml) and [infer.yaml](config/infer.yaml).
  

### Infer with pre-trained agents
1. Choose the sequence to replay. 
   Change the sequence name (**seq_name**) in [infer.yaml](config/infer.yaml)
2. Change the name of pretrained models. 
   Loaded by PyTorch in [infer.py](infer.py)
3. Run the inference
```
  python infer.py
```

### Train customized agents
1. Customize the parameters in [train.py](train.py)
2. Run the training
```
  python train.py
```

## Results
**Corridor**
<table>
  <tr>
    <td align="center"><img src="docs/corridor/built-in.gif" width="100" alt="built-in"><br>Built-in</td>
    <td align="center"><img src="docs/corridor/shim.gif" width="100" alt="shim"><br>Shim</td>
    <td align="center"><img src="docs/corridor/drl-stat.gif" width="100" alt="drl-stat"><br>DRL-stat</td>
    <td align="center"><img src="docs/corridor/drl-feat.gif" width="100" alt="drl-feat"><br>DRL-Feat</td>
  </tr>
</table>

**Parking**
<table>
  <tr>
    <td align="center"><img src="docs/parking/built-in.gif" width="100" alt="built-in"><br>Built-in</td>
    <td align="center"><img src="docs/parking/shim.gif" width="100" alt="shim"><br>Shim</td>
    <td align="center"><img src="docs/parking/drl-stat.gif" width="100" alt="drl-stat"><br>DRL-stat</td>
    <td align="center"><img src="docs/parking/drl-feat.gif" width="100" alt="drl-feat"><br>DRL-Feat</td>
  </tr>
</table>

## License

The source code is released under [MIT](https://opensource.org/license/MIT) license.
