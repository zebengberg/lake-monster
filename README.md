# lake-monster

> Use reinforcement learning to solve the lake monster problem.

- [Introduction](#introduction)
- [Installation and Usage](#installation-and-usage)
- [Environment](#environment)
- [Results](#results)
- [License](#license)

## Introduction

You find yourself in the middle of a circular lake in a rowboat. Glancing toward the shore, you spot a monster watching your every move. You start to row away from the monster. It runs along the circumference, tracking you, always aiming for the point on the shore closest to you. You know that once you get to the shore, you'll be able to out run the monster. However, stuck on the water, the monster is clearly faster than you. If you pick the optimal path to the shore, is it possible to escape? More precisely, how many times faster than you can the monster move so that you can still find a way to escape?

This [lake-monster problem](http://datagenetics.com/blog/october12013/index.html) is a classic mathematical riddle involving basic geometry and calculus. Determining the optimal path to the shore and calculating the monster's maximal speed under which you can escape is tricky. The [shape of the optimal path](https://puzzling.stackexchange.com/a/2161) is complex, involving several distinct stages. In short, although the solution to the problem is well understood mathematically, it is difficult to describe with simple geometric motions. For these reasons, the lake-monster problems makes an excellent testing ground for reinforcement learning.

Reinforcement learning (RL) is a machine learning framework in which an _agent_ interacts with an _environment_ in hopes of maximizing some long-term _reward_. In RL, the agent observes its environment at discrete time steps. Using its decision making _policy_, the agent chooses an _action_ to take within the environment. The environment may change in response to the agent's action, resulting in a new observation at the following time step. This observation -- action cycle continues until some terminal environment state is encountered. The agent seeks to maximize the reward it obtains from its trajectory within the environment. One full trajectory is known as an _episode_.

The lake-monster problem readily adheres to the RL framework. The _environment_ consists of position of the monster and rowboat within the lake-shore geometry and the monster's speed. The _agent_ is the human in the rowboat, the _policy_ is the human's decision making process, and the _action_ is the direction in which to propel the rowboat. After an action is taken, the environment is updated as the monster runs across an arc of the lake's circumference in hopes of arriving at the human's intersection with the shore prior to the human arrival. The _episode_ is the sequence of actions resulting in the human's escape or capture.

TODO: include gif of trained here

## Installation and Usage

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

## Environment and Agent

The [environment](environment.py) module contains the `LakeMonsterEnvironment` class, defining the RL environment of the lake-monster problem. The `LakeMonsterEnvironment` inherits from `PyEnvironment`, an abstract base class within the tf-agents package used for building custom environments.

The [agent](agent.py) module contains the `Agent` class, defining the DQN-agent which seeks to maximize its reward within the environment. The `Agent` class contains instances of many tf-agent objects used throughout the training process. Instance variables include training and evaluation environments (wrapped into TensorFlow objects), a replay buffer for enabling agent training, and several other objects needed within the tf-agent train data pipeline.

We represent the lake as a unit circle, the monster as a point on the circumference of that unit circle, and the agent as a point on the interior of the unit circle. Although the lake-monster problem deals with continuous motion, we discretize the problem in order to simulate it in a digital setting. In this implementation, discretizing the problem only gives additional advantages to the monster. After every step taken by the agent, the monster takes its own step. After taking a final step beyond the shoreline, the monster has one last chance to catch and eat the agent.

To keep episodes of the environment finite, we impose a maximum number of steps that the agent is allowed to take. Once the agent exceed this threshold, the current episode terminates.

### State

The state of the environment can be encapsulated in a single vector describing the position of the monster, the position of the agent, and the number of steps taken with the episode. Other parameters, such as the speed of the monster, the step size, and the space of possible agent actions could also be included within the state.

To account for symmetries of the circle, after each time step, we rotate the entire lake (agent and monster included) so that the monster is mapped to the point (1, 0). This, hopefully, will reduce the complexity of the problem for the agent.

In this particular implementation, our state is the 6-dimensional vector containing the following variables.

- the ratio of the current step count to the maximum allowed step count
- the monster's speed (constant throughout episode)
- the x and y Cartesian-coordinates of the agent
- the r and theta polar-coordinates of the agent

While the Cartesian and polar coordinates give redundant information, the neural network underlying the agent's policy may benefit from it.

### Action

Currently, the tf-agent implementation of a DQN agent requires the action of the agent to be both 1-dimensional and discrete. Consequently, the action of the environment is simply an angle corresponding to the direction in which the agent moves. We wrap `LakeMonsterEnvironment` with the tf-agent class `ActionDiscretizeWrapper` to achieve the discretization. The number of possible directions can be passed as `num_actions` to the `Agent` class and has a strong effect on the complexity of the Q-network underlying the agent's policy.

### Reward

The agent receives a positive reward if it successfully escapes from the lake without being captured by the monster. The agent receives a negative reward if it is captured, or if the number of steps exceeds the maximum allowable steps within an episode.

### Agent training and evaluation

greedy vs exploration

## Results

There are a number of hyperparameters involved in both the environment specification as well as the DQN agent's network. We describe significant parameters affecting agent performance here.

- `num_actions`
- `step_size`
- `initial_monster_speed`
- `timeout_factor`
- `hidden_layer_nodes`

### Nontrival results

Suppose the agent runs antipodally away from the monster. In turn, the monster will traverse the lake circumference, aiming for the point at which the agent will intersect with the shoreline. (The monster could make his semi-circumnavigation in a clockwise or counterclockwise motion; both paths will take the same time.) In this episode, the agent will travel a total distance of 1 whereas the monster will travel a total distance of pi. Therefore, the agent will succeed with this strategy if and only the speed of the monster is less than pi.

Because this strategy is so simple, we consider it as a baseline minimum for what the agent should aspire to learn. In other words, the agent has learned something nontrivial if it can escape from a monster who has a speed of at least pi. Due to the discretized nature of the environment (specifically, the discrete variables `step_size` and `num_actions`), the agent would not be able to enact this exact strategy. Nevertheless, we still consider a monster speed of pi as a baseline for an intelligent agent.

### Training wheels, passoffs, units

The lake-monster problem exhibits a _monotonicity_ property which we can hope to leverage in training.

## License

[MIT License](LICENSE.md)
