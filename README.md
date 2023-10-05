# AdMVDet

## Contents
- [AdMVDet](#admvdet)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [CARLA environment preparation](#carla-environment-preparation)
  - [Code Execution](#code-execution)
    - [Preparation](#preparation)
    - [Phase 1: Multiview detector training](#phase-1-multiview-detector-training)
    - [Phase 2: Reinforcement learning agent training](#phase-2-reinforcement-learning-agent-training)
    - [Phase 3: Detector output heads fine-tuning](#phase-3-detector-output-heads-fine-tuning)
    - [Conducting experiments in parallel](#conducting-experiments-in-parallel)

## Overview
Adaptive Multiview Detection, in short, AdMVDet, refers to the network framework that enables camera controls to adaptively detect pedestrians in the scene. The multiview detector is powered by the MVDet model developed by [Hou *et al.* (2020)](https://link.springer.com/chapter/10.1007/978-3-030-58571-6_1), and the adaptive controller agent is trained using the reinforcement learning algorithm.

The source code in this repository is written for the purpose of my 2023 Honours project at the Australian National University (ANU).

In addition, credits to the [MVSelect](https://github.com/hou-yz/MVSelect) repository by [Yunzhong Hou](https://github.com/hou-yz), from which this repository is developed.

## Prerequisites
To run the code, the machine MUST have a number of GPUs that are both greater than or equal to two and a multiple of two, with CUDA support. The code has been tested on Ubuntu 18.04.3 with Python 3.8.17 and CUDA 11.6.

To install the required Python libraries, execute the code below.
```
pip3 install -r requirements.txt
```

## CARLA environment preparation
To run the code, the CARLA simulator must be installed. The code has been tested on CARLA 0.9.14. We use the CARLA docker image to run experiments. You need to first install the docker engine and configure the Linux user group as documented in this [instruction](https://docs.docker.com/engine/install/ubuntu/). Then, install the Nvidia container toolkit following this [document](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Then, you can pull the CARLA docker image by executing the code below.

```
docker pull carlasim/carla:0.9.14
```

All done! You can try to run the CARLA simulator by executing the code below.

```
docker run -d --privileged --gpus "device=all" --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -RenderOffScreen
```

## Code Execution
In this section, we will introduce how to generate the configuration file and run the code step by step. You are very welcome to modify other scripts like `dataset_stats.py`, `plot_rl_reward.py`, `result_parser.py`, and `run.py` to accommodate your needs, which are useful to run experiments batch-wise and analyze the results in my thesis writing. Besides, I have several Jupyter notebook files that are not included in this repository. If you are interested in them, please contact me via email at xingjian.leng@anu.edu.au.

### Preparation
In the first step, you need to generate the configuration file for the CARLAX dataset. You can do this by executing the code below.

```
python3 ./src/environment/create_config_rl.py
```

After obtaining the `1.cfg` file in the `./cfg/` directory, you can start a CARLA docker container by executing the command mentioned above, and proceed to the next step. Note that you are expected to use another GPU for model training.

### Phase 1: Multiview detector training
In the second step, you need to train the multiview detector. You can do this by executing the code below to train a detector with the uniform spawn strategy.

```
python main.py --cfg_path ./cfg/1.cfg -d carlax --spawn_strategy uniform --log_interval 50
```

### Phase 2: Reinforcement learning agent training

In the third step, you need to train the reinforcement learning agent. You can run the command below to train an agent with the uniform spawn strategy, the delta_moda reward, and the conv_base architecture variant.

```
python main.py --cfg_path ./cfg/1.cfg -d carlax --interactive --base_lr_ratio 0.0 --other_lr_ratio 0.0 --reward delta_moda --spawn_strategy uniform --rl_variant conv_base --log_interval 50
```

### Phase 3: Detector output heads fine-tuning
Finally, to fine-tune the output heads of the detector using a specific checkpoint, execute the command provided below. This command incorporates the uniform spawn strategy, the delta_moda reward mechanism, and the conv_base architecture variant. Make sure to specify the correct path to your phase 2 checkpoint directory.

```
python main.py --cfg_path ./cfg/1.cfg -d carlax --interactive --fine_tune --rl_variant conv_base --resume [path_to_phase2_checkpoint_folder] --spawn_strategy uniform --reward delta_moda --log_interval 50
```

### Conducting experiments in parallel
You can also run multiple experiments in parallel by starting another CARLA docker container with the command below. It can specify the port mapping of the CARLA server.

```
docker run -d --privileged --gpus "device=[another_gpu_id]" --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -RenderOffScreen -world-port=3000 -carla-rpc-port=4000
```

Additionally, you need to set both `--port 3000` and `--tm_port 5000` flags when executing the `main.py` file to enable the CARLA client to connect to the server.
