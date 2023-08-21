# GRADIENT: Curriculum Reinforcement Learning using Optimal Transport via Gradual Domain Adaptation

## Paper
Huang P, Xu M, Zhu J, Shi L, Fang F, Zhao D. Curriculum reinforcement learning using optimal transport via gradual domain adaptation. Advances in Neural Information Processing Systems. 2022 Dec 6;35:10656-70. https://arxiv.org/abs/2210.10195

## Install dependencies
```
conda create --name gradient python=3.8.12
pip install -r requirements.txt
```
```
cd envs/gym && pip install -e . 
```
```
cd envs/mujoco-maze && pip install -e . 
```

## Environments: 
- Environments are modified from Mujoco_maze (https://github.com/kngwyu/mujoco-maze) and gym (https://github.com/openai/gym).

## Code Usage
```
python  run_maze_continuous.py --curriculum gradient --interp_metric encoding --num_stage 5 --reward_threshold 0.5
python  run_maze_continuous.py --curriculum gradient --interp_metric l2 --num_stage 5 --reward_threshold 0.5
```