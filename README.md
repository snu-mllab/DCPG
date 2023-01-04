# Rethinking Value Function Learning for Generalization in Reinforcement Learning

[![DOI](https://zenodo.org/badge/548142480.svg)](https://zenodo.org/badge/latestdoi/548142480)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/snu-mllab/DCPG/blob/main/LICENSE) 

This is the code for reproducing the results of the paper [Rethinking Value Function Learning for Generalization in Reinforcement Learning](http://arxiv.org/abs/2210.09960)
accepted at NeurIPS 2022.

## Acknowledgement

> This work was supported in part by Samsung Advanced Institute of Technology, Samsung Electronics Co., Ltd., Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2019-0-01371, Development of brain-inspired AI with human-like intelligence, No. 2020-0-00882, Development of deployable learning intelligence via self-sustainable and trustworthy machine learning, and No. 2022-0-00480, Development of Training and Inference Methods for Goal-Oriented Artificial Intelligence Agents), and a grant of the Korea Health Technology R&D Project through the Korea Health Industry Development Institute (KHIDI), funded by the Ministry of Health & Welfare, Republic of Korea (grant number: HI21C1074). This material is based upon work supported by the Air Force Office of Scientific Research under award number FA2386-22-1-4010.


## Installation

To install all required dependencies, please run the following commands in the project root directory.

```
conda create —-name procgen python=3.8
conda activate procgen

pip install numpy==1.23.0
pip install tensorflow==2.9.0
conda install pytorch=1.11.0 cudatoolkit=11.3 -c pytorch

pip install procgen
pip install pyyaml
pip install -e .

git clone https://github.com/openai/baselines.git
cd baselines 
pip install -e .
```

If your GPU driver does not support CUDA 11.2 or later, please downgrade CUDA toolkit for PyTorch and TensorFlow.
Here are the recommended versions for CUDA 10.2.

```
pip install tensorflow==2.3.0
conda install pytorch=1.11.0 cudatoolkit=10.2 -c pytorch
```

## Usage

PPO (baseline)

```
python train.py --exp_name ppo --env_name [EVN_NAME]
```

DAAC (baseline)

```
python train.py --exp_name daac --env_name [EVN_NAME]
```

PPG (baseline)

```
python train.py --exp_name ppg --env_name [EVN_NAME]
```

DCPG (ours)

```
python train.py --exp_name dcpg --env_name [EVN_NAME]
```

DDCPG (ours)

```
python train.py --exp_name ddcpg --env_name [EVN_NAME]
```

You can find the corresponding config file for each experiment in the directory named `configs`. If you want to customize an experiment, please create a new config file named `[CONFIG_NAME].yaml` in the directory and run the following command.

```
python train.py --exp_name [CONFIG_NAME] --env_name [EVN_NAME]
```

## Citation

Please cite this work using the following bibtex entry.

```
@inproceedings{moon2022dcpg,
    title={Rethinking Value Function Learning for Generalization in Reinforcement Learning},
    author={Seungyong Moon and JunYoung Lee and Hyun Oh Song},
    booktitle={Neural Information Processing Systems},
    year={2022}
}
```

## Credit
- [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
- [https://github.com/rraileanu/idaac](https://github.com/rraileanu/idaac)
- [https://github.com/openai/phasic-policy-gradient](https://github.com/openai/phasic-policy-gradient)
