# lake-monster

> Use reinforcement learning to solve the lake monster problem.

- [Introduction](#introduction)
- [Install](#install)
- [Environment](#rl-environment)
- [tf-agents](#tf-agents)
- [Results](#results)
- [License](#license)

## Introduction

You find yourself in the middle of a circular lake in a rowboat. Glancing toward the shore, you spot a monster watching your every move. You start to row away from the monster. It runs along the circumference, tracking you, always aiming for the point on the shore closest to you. You know that once you get to the shore, you'll be able to out run the monster. However, stuck on the water, the monster is clearly faster than you. If you pick the optimal path to the shore, is it possible to escape? More precisely, how many times faster than you can the monster move so that you can still find a way to escape?

This [lake-monster problem](http://datagenetics.com/blog/october12013/index.html) is a classic mathematical riddle involving basic geometry and calculus. Determining the optimal path to the shore and calculating the monster's maximal speed under which you can escape is tricky. The [shape of the optimal path](https://puzzling.stackexchange.com/a/2161) is complex, involving several distinct stages. In short, although the solution to the problem is well understood mathematically, it is difficult to describe with simple geometric motions. For these reasons, the lake-monster problems makes an excellent testing ground for reinforcement learning.

Reinforcement learning (RL) is a machine learning framework in which an _agent_ interacts with an _environment_ in hopes of maximizing some long-term _reward_. In RL, the agent observes its environment at discrete time steps. Using its decision making _policy_, the agent chooses an _action_ to take within the environment. The environment may change in response to the agent's action, resulting in a new observation at the following time step. This observation -- action cycle continues until some terminal environment state is encountered. The agent seeks to maximize the reward it obtains from its trajectory within the environment.

The lake-monster problem readily adheres to the RL framework. The _environment_ consists of position of the monster and rowboat within the lake-shore geometry and the monster's speed. The _agent_ is the human in the rowboat, the _policy_ is the human's decision making process, and the _action_ is the direction in which to propel the rowboat. After an action is taken, the environment is updated as the monster runs across an arc of the lake's circumference in hopes of arriving at the human's intersection with the shore prior to the human arrival.

TODO: include gif of trained here

## Install

This simulation was created with Python 3.8 using TensorFlow and TF-Agents.

You can clone or download this repository locally. You may want to create a new Python environment to install the dependencies. This can be accomplished with `conda`.

```sh
conda create --name monster python=3.8
conda activate monster
pip install -r requirements.txt
```

Test the lake-monster environment by running `python test_environment.py`. This script will initialize an agent with a random policy to interact with several versions of `LakeMonsterEnvironment`. A test video will also be generated showing the random movement of the agent within the environment.

![random policy](assets/random.gif)

Train a DQN agent to learn to solve the lake-monster puzzle by running `python acquire_knowledge.py`. This module will periodically save several pieces training knowledge.

- TensorFlow checkpoints, such as weights of the Q-learning neural network, are saved as binary files in the `checkpoints/` directory.
- Training and environment statistics, such as reward, training loss, and number of environment steps, are saved in `stats.json`. Running `python stats.py` will display plots of the generated statistics.
- Video renders of the agent interacting with the environment are periodically captured and stored within the `videos/` directory.

Training can be keyboard interrupted at any point in training and continued from a saved checkpoint later on. This is handled automatically through the tf-agent `checkpointer` object and the `stats` module. Training hyperparameters and environment parameters can be directly specified in the `acquire_knowledge` script. See [Results](#results) for a discussion of parameter selection.

Training knowledge can be completely reset by running `python clear_knowledge.py`.

## Environment

The [environment](environment.py) module contains the `LakeMonsterEnvironment` class, which defines the RL environment of the lake-monster problem.

## tf-agents

## Results

There are a number of hyperparameters involved in both the environment specification as well as the DQN agent's network. We describe significant parameters affecting agent performance here.

### Training wheels

### pi as threshold

## License

[MIT License](LICENSE.md)
