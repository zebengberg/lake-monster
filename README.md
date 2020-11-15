# lake-monster

> Use reinforcement learning to solve the lake monster problem.

- [Introduction](#introduction)
- [Installation and Usage](#installation-and-usage)
- [Environment and Agent](#environment-and-agent)
- [Learning](#learning)
- [Results](#results)
- [License](#license)

## Introduction

You find yourself in the middle of a circular lake in a rowboat. Glancing toward the shore, you spot a monster watching your every move. You start to row away from the monster. It runs along the circumference, tracking you, always aiming for the point on the shore closest to you. You know that once you get to the shore, you'll be able to out run the monster, avoiding death. However, stuck on the water, the monster is clearly faster than you. If you row in an optimal path to the shore, is it possible to escape? More precisely, how many times faster than you can the monster move so that you can manage to escape?

|                ![capture](assets/capture.gif)                 |
| :-----------------------------------------------------------: |
| _An agent who cannot quite escape and eventually gets eaten._ |

This [lake-monster problem](http://datagenetics.com/blog/october12013/index.html) is a classic mathematical riddle involving basic geometry and calculus. It is tricky to determine the optimal path to the shore and the maximal monster speed under which escape is possible. The [shape of the optimal path](https://puzzling.stackexchange.com/a/2161) is complex, involving several distinct stages. In short, although the solution to this problem is well understood mathematically, it is difficult to describe in simple geometric motions. For these reasons, the lake-monster problems provides a fertile testing ground for reinforcement learning algorithms.

Reinforcement learning (RL) is a machine learning framework in which an _agent_ interacts with an _environment_ in hopes of maximizing some long-term _reward_. In RL, the agent observes its environment at discrete time steps. Using its decision making _policy_, the agent chooses an _action_ to take within the environment. The environment updates in response to the agent's action, resulting in a new observation at the following time step. This observation-action cycle continues until some terminal environment state is encountered. The agent seeks to maximize the reward it obtains from interacting the environment. One full trajectory from an initial state to a terminal state is known as an _episode_.

The lake-monster problem readily adapts to the RL framework. The _environment_ consists of the rowboat situated within the lake and the monster running on the shore. The _agent_ is the human guiding the rowboat hoping to escape, the _policy_ is the human's decision making process, and the _action_ is the direction in which the human chooses to propel the rowboat. After an action is taken, the environment is updated as the monster runs across an arc of the lake's circumference in hopes getting closer to the tasty human. The _episode_ is the sequence of human actions resulting in the human's escape or capture.

## Installation and Usage

### Installation

This simulation was created with Python 3.8 using TensorFlow and TF-Agents.

You can clone or download this repository locally. You may want to create a new Python environment to install the dependencies using an environment manager such as `conda`.

```sh
conda create --name monster python=3.8
conda activate monster
pip install -r requirements.txt
```

### Tests

Test the lake-monster environment by running `python test_environment.py`. This script will initialize a random policy to interact with `LakeMonsterEnvironment` (see [Environment](#environment-and-agent)) and create a sample video showing an episode. Test the TensorFlow-derived DQN-based agent framework by running `python test_agent.py`. This script will instantiate several `Agent` objects, print out network statistics, and run a few episodes using the TF-agent `dynamic_episode_driver` pipeline.

|                ![random policy](assets/random.gif)                |
| :---------------------------------------------------------------: |
| _An agent with a random policy interacting with the environment._ |

### Usage

After testing the basic components, you can train your very own agent to solve the lake-monster puzzle by running `python acquire_knowledge.py`. Default parameters are strong, and can be modified in the source code. See [Results](#results) for a discussion of parameter selection. As the agent learns, several pieces training knowledge are saved.

- TensorFlow checkpoints are saved in the `checkpoints/` directory. These checkpoints hold the current state of the agent and its associated TF-agent objects (such as the _replay buffer_). Training can be keyboard interrupted at any point in time and continued from the last saved checkpoint by re-running `acquire_knowledge.py`.
- Training statistics are logged through `tf.summary` methods and saved in the `logs/` directory. Statistics are indexed by the episode number and include reward, training loss, number of environment steps taken by the agent, and network weights. These statistics can be viewed through TensorBoard (see images below), which automatically loads in a browser tab. In addition to displaying responsive plots, TensorBoard provides some tools for more detailed data analysis, such as the ability to export a time series as a CSV file.
- Video renders of the agent interacting with the environment are periodically captured and stored within the `videos/` directory.
- Agent policies are periodically time-stamped and saved in the `policies/` directory. Saved policies are slimmed-down checkpoints containing only hyperparameter values and network weights. They cannot be used for continuing agent training; instead, they demonstrate the actions of a learned policy.
- Results across distinct agents are saved in `results.json`. The file `agent_id.txt` holds a UUID corresponding to a partially trained agent.

Training knowledge can be reset by running `python clear_knowledge.py`. This script will clear the `checkpoints/`, `logs/`, and `videos/` directories, and remove the `agent_id.txt` file. This script should be run when training a new agent from scratch. Saved policies and items in `results.json` are not removed.

|                 ![loss](assets/loss.png)                 |             ![weights](assets/weights.png)             |
| :------------------------------------------------------: | :----------------------------------------------------: |
| _A loss statistic tracked over training in TensorBoard._ | _Offset histograms of the weights of a network layer._ |

## Environment and Agent

The [environment](environment.py) module contains the `LakeMonsterEnvironment` class which defines the RL environment of the lake-monster problem. The `LakeMonsterEnvironment` inherits from `PyEnvironment`, an abstract base class within the TF-agents package used for building custom environments.

The [agent](agent.py) module contains the `Agent` class. This defines the DQN-agent which seeks to maximize its reward within the environment. The `Agent` class contains instances of TF-agent objects used throughout the training process. Instance variables include training and evaluation environments (wrapped into TensorFlow objects), a replay buffer for enabling DQN learning, and several other objects needed within the TF-agent training pipeline.

We represent the lake as a unit circle, the monster as a point on the circumference of that unit circle, and the agent as a point in the interior of the unit circle. Although the lake-monster problem deals with continuous motion, we discretize the problem in order to simulate it in a DQN-setting. In this implementation, discretizing the problem only gives additional advantages to the monster: the agent has less freedom in its motion. After every step taken by the agent, the monster takes its own step. After taking a final step beyond the shoreline, the monster has one final chance to capture and eat the agent.

At each time step, the agent has only finitely many actions to choose among (see [Action](#action)). Additionally, we impose a maximum number of steps that the agent is allowed to take. Once the agent exceed this threshold, the current episode terminates. Consequently, there are finitely many trajectories that the agent could take within the environment.

### State

The state of the environment can be encapsulated in a single vector describing the position of the monster, the position of the agent, and the number of steps taken with the current episode. Other parameters, such as the speed of the monster, the step size, and the space of possible agent actions could also be included within the state. We experiment with several different choices of the `_state` vector within the `LakeMonsterEnvironment` class. In particular, we try both polar and Cartesian coordinates to describe the position of the agent.

To benefit from the symmetries of the circle, after each time step we rotate the entire lake (agent and monster included) so that the monster is mapped to the point (1, 0). We hope that this transformation reduces the complexity of the problem for the agent.

### Action

Currently, the TF-agent implementation of a DQN agent requires the action of the agent to be both 1-dimensional and discrete. The parameter `num_actions` specifies the number of possible directions that agent can step. We map each integer action to an angle uniformly distributed over the circle. The parameter `num_actions` is the dimension of the output of the policy network. Choosing a small value for `num_actions` (such as 4) will enable faster learning at the expense of less mobility for the agent.

|             ![discrete actions](assets/actions.gif)              |
| :--------------------------------------------------------------: |
| _An agent with `num_actions = 8` moving around the environment._ |

### Reward

The agent receives a reward of +1 if it successfully escapes from the lake without being captured by the monster and receives no reward if it is captured. If the agent takes more steps than allowed within an episode, the agent receives a small negative reward. The purpose of discouraging timeouts is to incentivize the agent to attempt an escape rather than wander aimlessly.

In some instances, we provide mini-rewards to our agent. Rather than give our agent a binary reward of 0 or 1, we give partial rewards for a near escape. If our the agent is captured by the monster, there must be a step within the episode at which the monster is at the agent's radial projection onto the circle. In other words, there must be a step at which the monster is situated on the shore waiting for the agent to arrive. An agent that escapes will never encounter a waiting monster, and so an agent who can make it closer to the shore before encountering a waiting monster is closer to escape. In this way, we can provide agents with larger rewards when they make it closer to the shore before finding a waiting monster. In a similar manner, we give agents additional rewards for succeeding by a wider margin. This can be quantified by considering the arc between the agent and the monster on the shore of the lack at the instant of escape.

The lake-monster problem is an example of a sparse reward or delayed-reward environment. Agents only receive rewards at the termination of an episode; there is no reward at a transitional step within each episode. Some episodes may take dozens or even hundreds of steps to finish; consequently, learning in such an environment is difficult. Although it is possible to craft transitional rewards for an agent to steer it toward an optimal path, doing so defeats the goal of autonomous agent learning. We avoid human crafted step by step rewards.

|                                                                                     ![rewards](assets/rewards.gif)                                                                                      |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| _An agent receiving a partial reward for getting closer to the shore. The velocity vector in the upper left hand corner of the animation changes color once the monster becomes inline with the agent._ |

## Learning

### Parameters

There are a number of parameters and hyperparameters involved in both the environment specification as well as the DQN agent. When initializing an `Agent` object, we specify the following variables.

- `num_actions` -- the number of possible directions the agent can move within the environment
- `initial_step_size` -- the length of each step made by the agent
- `initial_monster_speed` -- the speed of the monster at the start of training
- `timeout_factor` -- the episode terminates once the number of steps exceeds `timeout_factor / step_size`
- `use_mini_rewards` -- see [Reward](#reward)
- `use_cartesian` -- in addition to polar coordinates, use Cartesian coordinates in environment state
- `use_noisy_start` -- rather than start at (0, 0), choose a random starting location for the agent
- `fc_layer_params` -- parameters for the fully connected neural network underpinning the policy
- `dropout_layer_params` -- used to create dropout layers in the neural network
- `learning_rate` -- a parameter for the neural network Adam optimizer
- `epsilon_greedy` -- a parameter to control the amount of exploration in DQN training
- `n_step_update` -- the number of steps to look ahead when computing loss of Q-values
- `use_categorical` -- see the TensorFlow [tutorial](https://www.tensorflow.org/agents/tutorials/9_c51_tutorial)
- `use_step_schedule`, `use_learning_rate_schedule`, `use_mastery` -- see [Modifying Parameters](#modifying-parameters)

As `num_actions` and `timeout_factor` grow large and `step_size` approaches 0, the discrete lake-monster environment approaches the idealized lake-monster continuous-motion environment. There is a natural trade-off as the discrete environment tends towards a continuous-motion environment. As `num_actions` grows, the output dimension of the neural network grows, thereby increasing the overall complexity of the network. As a result, the policy will take longer to train. Additionally, DQN exploration will grow linearly in `num_actions`. As `step_size` becomes small, the agent will make slower progress toward reaching the shoreline with untrained movements. Each episode will take longer to run, and the entire training process will slow. As `timeout_factor` increases, an agent may spend additional time wandering without progress thereby slowing learning. Conversely, as the environment tends toward one of continuous-motion, the agent is afforded more freedom in its movement which can allow it to perform better after extensive training.

The lake-monster problem is provides a forgiving environment for the agent. If the agent makes mistakes in its actions, it has the ability to correct them by taking actions to bring the environment close to its initial state. In the optimal solution, an agent would never arrive at a position radially inline with the monster. In the gif below, this occurs when the velocity vector changes color. The agent retreats from this sub-optimal state before continuing on to eventually succeed.

|             ![rewards](assets/path.gif)             |
| :-------------------------------------------------: |
| _A well-trained agent correcting initial missteps._ |

### Modifying Parameters

Some of the parameters discussed above cannot be tweaked without instantiated a new agent. In general, parameters required to initialize the neural network or replay buffer underpinning the agent cannot be changed. These include `num_actions`, `use_cartesian`, `n_step_update`, `use_cateorical`, and any of the layer parameters.

On the other hand, some parameters can be modified without changing the basic structure of the learned objects. These include `monster_speed`, `step_size`, and `learning_rate`. The lake-monster problem exhibits a _monotonicity_ property which we can hope to leverage in training. If an agent can succeed against a monster with a high speed, by taking the same path, it can succeed against a monster with a lower speed. We expect this monotonicity to also hold with the `step_size` parameter: if an agent can escape from a monster when taking large steps, we also expect it to be able to escape from a monster by taking many smaller steps.

When training an agent, we could use a _mastery-based_ approach. In such a setting, an agent initially encounters a slow monster. Once the agent has shown mastery in this easier environment, the speed of the monster is increased. In this way, the agent is always paired with a monster whose speed perfectly matches the ability of the agent.

Extending this idea, the parameters `use_step_schedule`, `use_learning_rate_schedule`, and `use_mastery` modify parameters over the course of training. Often, this leads to better outcomes. See [Results](#results) for a discussion.

Agent policies are periodically saved (see [Usage](#usage)) for later use. A well trained policy can be revived and evaluated with modified parameters. The agents who received extensive training often succeed when paired monsters with higher speeds simply by placing these agents into a more continuous environment.

|                                                                                                         ![many agents](assets/many.gif)                                                                                                         |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| _A collection of agents in various states of training. Here, the monster is fixed in place and all agents are rotated toward the monster to account for its motion. Agents that stick to the wall above the monster have successfully escaped._ |

### Nontrival Learning

Suppose the agent runs antipodally away from the monster. The monster will traverse the lake circumference, aiming for the point at which the agent will intersect with the shoreline. (The monster could make its semi-circumnavigation in a clockwise or counterclockwise motion; both paths will lead to identical outcomes.) In such an episode, the agent will travel a total distance of 1 whereas the monster will travel a total distance of pi. Therefore, the agent will succeed with this strategy if and only the speed of the monster is less than pi.

Because this strategy is so simple, we consider it as a baseline minimum for what the agent should aspire to learn. In other words, the agent has learned something nontrivial if it can escape from a monster who has a speed of at least pi. Due to the discretized nature of the environment (specifically, the discrete variables `step_size` and `num_actions`), the agent would not be able to enact this exact strategy. Nevertheless, we still consider a monster speed of pi as a baseline for an intelligent agent.

With no knowledge of the mathematics, it is possible for a human to quickly experiment and do better. For example, the function `test_movement` in the [`test_environment` module](test_environment.py) uses simple handcrafted actions to succeed against a monster with speed 3.5. In the lake-monster problem, we expect most humans could eventually fine-tune a set of actions to succeed against a monster with speeds between pi and 4.0. We consider an agent to have _human-level_ intelligence if it can succeed against monsters with speeds up to 4.0. An agent who can escape from a monster with speeds above 4.0 is considered to have _super-human-level_ intelligence.

Of course, we know that the [optimal mathematical solution](http://datagenetics.com/blog/october12013/index.html) allows an agent to escape a monster with speed 4.6. While speeds of this high are not possible in this discrete version of the problem, we expect to be able to get close to this upper bound.

|           ![strong policy](assets/strong.gif)            |
| :------------------------------------------------------: |
| _An agent demonstrating highly complex learned actions._ |

## Results

In progress ....

## License

[MIT License](LICENSE.md)
