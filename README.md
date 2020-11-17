# lake-monster

> Use reinforcement learning to solve the lake monster problem.

- [Introduction](#introduction)
- [Installation and Usage](#installation-and-usage)
- [Environment and Agent](#environment-and-agent)
- [Learning](#learning)
- [Results](#results)
- [License](#license)

## Introduction

You find yourself in the middle of a circular lake in a rowboat. Glancing toward the shore, you spot a monster watching your every move. Startled, you start to row away from the monster. It runs along the circumference, tracking you, always heading toward the point on the shore closest to you. If you can get to the shore without intersecting the monster, you'll be able to outrun it, avoiding death. However, stuck in your rowboat, the monster is clearly faster than you. If you row in an optimal path to the shore, is it possible to escape? More precisely, what is the maximum speed at which the monster can move while still providing you with the opportunity to escape?

|                ![capture](assets/capture.gif)                 |
| :-----------------------------------------------------------: |
| _An agent who cannot quite escape and eventually gets eaten._ |

This [lake-monster problem](http://datagenetics.com/blog/october12013/index.html) is a classic mathematical riddle involving basic geometry and calculus. It is difficult to determine the optimal path to the shore and the maximal monster speed under which escape is possible. The [shape of the optimal path](https://puzzling.stackexchange.com/a/2161) is complex, involving both spiral and linear motion. In short, although the solution to this problem is well understood mathematically, it is difficult to discover without mathematical insight. It is unlikely that a human player would be able to stumble across a near-optimal solution through unguided trial and error. For these reasons, the lake-monster problem provides a novel and fertile testing ground for reinforcement learning algorithms.

Reinforcement learning (RL) is a machine learning framework in which an _agent_ interacts with an _environment_ in hopes of maximizing some long-term _reward_. In RL, the agent observes its environment at discrete time steps. Using its decision making _policy_, the agent chooses an _action_ to take within the environment. The environment updates in response to the agent's action, resulting in a new observation at the following time step. This observation-action cycle continues until some terminal environment state is encountered. One full trajectory from an initial state to a terminal state is known as an _episode_.

The lake-monster problem is readily adaptable to the RL framework. Specifically, the _environment_ consists of the lake, the rowboat, and the land-locked monster. The _agent_ is guiding the rowboat hoping to escape. The _policy_ is the agent's decision making process. The _action_ is the direction in which the agent chooses to propel the rowboat. After an action is taken, the environment is updated as the monster runs across an arc of the lake's circumference in hopes of getting closer to the tasty agent. The _episode_ is the sequence of agent actions resulting in the agent's escape or capture. The RL algorithm used in this project is the Deep Q-Network (DQN) algorithm.

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

Test the lake-monster environment by running `python test_environment.py`. This script initializes a random policy to interact with `LakeMonsterEnvironment` (see [Environment](#environment-and-agent)) and create a sample video showing an episode. Test the TensorFlow-derived DQN-based agent framework by running `python test_agent.py`. This script instantiates several `Agent` objects, print out network statistics, and run a few episodes using the TF-Agents `DynamicEpisodeDriver` pipeline.

|                ![random policy](assets/random.gif)                |
| :---------------------------------------------------------------: |
| _An agent with a random policy interacting with the environment._ |

### Usage

After testing the basic components, you can train your very own agent to solve the lake-monster puzzle by running `python acquire_knowledge.py`. Default parameters are strong, and can be modified in the source code. See [Learning](#learning) for a discussion of parameter selection. As the agent learns, several pieces of training knowledge are saved.

- TensorFlow checkpoints are saved in the `checkpoints/` directory. These checkpoints store the current state of the agent and its associated TF-Agents objects (for example, the _replay buffer_). Training can be keyboard interrupted at any point in time and continued from the last saved checkpoint by re-running `acquire_knowledge.py`.
- Training statistics are logged through `tf.summary` methods and saved in the `logs/` directory. Statistics are indexed by the episode number and include metrics such as reward, training loss, number of environment steps taken by the agent, and network weights. These statistics can be viewed through TensorBoard (see images below), which automatically loads in a browser tab.
- Video renders of the agent interacting with the environment are periodically captured and stored within the `videos/` directory.
- Agent policies are periodically time-stamped and saved in the `policies/` directory. Saved policies are slimmed-down checkpoints containing only hyperparameter values and network weights. They are used to evaluate agents and render episodes after training is complete.
- Results across distinct agents are saved in `results.json`.

Training knowledge can be reset by running `python clear_knowledge.py`. This script clears the `checkpoints/`, `logs/`, and `videos/` directories, and removes the `agent_id.txt` file. This script should be run before training a new agent from scratch. Saved policies and items in `results.json` are not removed.

|                 ![loss](assets/loss.png)                 |             ![weights](assets/weights.png)             |
| :------------------------------------------------------: | :----------------------------------------------------: |
| _A loss statistic tracked over training in TensorBoard._ | _Offset histograms of the weights of a network layer._ |

## Environment and Agent

The [environment](environment.py) module contains the `LakeMonsterEnvironment` class which defines the RL environment of the lake-monster problem. `LakeMonsterEnvironment` inherits from `PyEnvironment`, an abstract base class within the TF-Agents package used for building custom environments.

The [agent](agent.py) module contains the `Agent` class. This defines the DQN agent which explores within the environment. An `Agent` object contains instances of TF-Agents objects used throughout the training process. Member variables include training and evaluation environments (wrapped into TensorFlow objects), a replay buffer for enabling DQN learning, and several other objects needed within the TF-Agents training pipeline.

We represent the lake as a unit circle, the monster as a point on the circumference of that unit circle, and the agent as a point in the interior of the circle. Even though the lake-monster problem deals with continuous motion, it can be discretized in order to simulate it in a DQN setting. Discretizing the lake-monster environment gives additional advantages to the monster since the agent is constrained in its movement.

At each time step, the agent has only finitely many actions to choose among (see [Action](#action)). Additionally, we impose a maximum number of steps that the agent is allowed to take. Once the agent exceeds this threshold, the current episode terminates. Consequently, there are finitely many trajectories (albeit an astronomically huge number) that the agent could take within the environment.

### State

The state of the environment can be encapsulated with a single vector that describes the position of the monster, the position of the agent, and the number of steps taken within the current episode. Other parameters, such as the speed of the monster, the step size, and the space of possible agent actions could also be included in the state. We experiment with several different choices of the `_state` vector within the `LakeMonsterEnvironment` class. In particular, both polar and Cartesian coordinates can be used to describe the position of the agent.

To take advantage of the symmetries of the circle, the entire lake (agent and monster included) is rotated after each time step so that the monster is mapped to the point (1, 0). This transformation reduces the complexity of the problem for the agent.

### Action

Currently, the TF-Agents implementation of a DQN agent requires the action of the agent to be both 1-dimensional and discrete. The parameter `num_actions` specifies the number of possible directions that the agent can step. Each integer action is mapped to a unique angle on the circle. The parameter `num_actions` is the dimension of the output of the policy network. Choosing a small value for `num_actions` (such as 4) enables faster learning at the expense of less mobility for the agent.

|             ![discrete actions](assets/actions.gif)              |
| :--------------------------------------------------------------: |
| _An agent with `num_actions = 8` moving around the environment._ |

### Reward

The agent receives a reward of +1 if it successfully escapes from the lake without being captured by the monster and receives no reward if it is captured. If the agent takes more steps than allowed within an episode, the agent receives a small negative reward. The purpose of discouraging timeouts is to incentivize the agent to attempt an escape rather than to wander aimlessly.

In some instances, the agent is provided with mini-rewards. Rather than give the agent a binary reward of 0 or 1, we give partial rewards for a near escape. If the agent is captured by the monster, there must be at least one step at which the monster is situated on the shore waiting for the agent to arrive. The longer an agent can avoid a waiting monster, the closer it is to escaping. For this reason, agents can be given with additional rewards when they make it close to the shore without encountering a waiting monster. Similarly, agents can be given bonus rewards for succeeding by a wider margin.

The lake-monster problem is an example of a _sparse reward_ or _delayed reward_ environment. Agents only receive rewards at the termination of an episode; there is no reward at a transitional step within an episode. Some episodes may take dozens or even hundreds of steps to finish; consequently, learning in such an environment is difficult. Although it is possible to craft transitional rewards that steer an agent toward an optimal path, doing so defeats the goal of autonomous human-independent agent learning. We avoid human crafted step-by-step rewards.

|                                                                                    ![mini reward](assets/reward.gif)                                                                                    |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| _An agent receiving a partial reward for getting closer to the shore. The velocity vector in the upper left hand corner of the animation changes color once the monster becomes inline with the agent._ |

## Learning

### Parameters

There are a number of parameters and hyperparameters involved in both the environment specification as well as the DQN agent. When initializing an `Agent` object, the following variables can be specified.

- `num_actions` -- set the number of possible directions the agent can step
- `initial_step_size` -- set the length of each step made by the agent
- `initial_monster_speed` -- set the speed of the monster at the start of training
- `timeout_factor` -- an episode terminates once the number of steps exceeds `timeout_factor / step_size`
- `use_mini_rewards` -- see [Reward](#reward)
- `use_cartesian` -- use Cartesian coordinates in the environment state
- `use_noisy_start` -- choose a random starting location for the agent rather than the origin
- `fc_layer_params` -- specify parameters for the fully connected neural network underpinning the policy
- `dropout_layer_params` -- create dropout layers in the neural network
- `learning_rate` -- set the learning rate for the neural network Adam optimizer
- `epsilon_greedy` -- control the amount of exploration in DQN training
- `n_step_update` -- the number of steps to look ahead when computing loss of Q-values
- `use_categorical` -- see this TensorFlow [tutorial](https://www.tensorflow.org/agents/tutorials/9_c51_tutorial)
- `use_step_schedule` and `use_mastery` -- see [Modifying Parameters](#modifying-parameters)

As `num_actions` and `timeout_factor` grow large and `step_size` approaches 0, the discrete lake-monster environment approaches the idealized lake-monster continuous-motion environment. There is a natural trade-off as the discrete environment tends towards a continuous-motion environment. As `num_actions` grows, the output dimension of the neural network grows, thereby increasing the overall complexity of the network. Additionally, DQN exploration grows linearly in `num_actions`. As a result, the policy takes longer to train. As `step_size` becomes small, the agent makes slower progress toward reaching the shoreline. Each episode takes longer to run, and the entire training process slows. As `timeout_factor` increases, an agent may spend additional time wandering without progress, thereby slowing learning. Conversely, as the environment tends toward one of continuous-motion, the agent is afforded more freedom in its movement which can allow it to perform better after extensive training.

The lake-monster problem provides a forgiving environment for the agent. If the agent makes poor decisions, it has the ability to correct them by taking actions to bring the environment back to its initial state. In the optimal solution, an agent would never arrive at a position radially inline with the monster. In the gif below, this occurs when the velocity vector changes color. The agent retreats from this sub-optimal state before continuing on to eventually succeed.

|             ![rewards](assets/path.gif)             |
| :-------------------------------------------------: |
| _A well-trained agent correcting initial missteps._ |

### Modifying Parameters

Some of the parameters discussed above cannot be tweaked without instantiating a new agent. In general, parameters used to initialize the neural network or replay buffer cannot be changed. These include `num_actions`, `use_cartesian`, `n_step_update`, `use_cateorical`, and any of the layer-specific parameters.

On the other hand, some parameters can be modified without changing the structure of the learned objects. These include `monster_speed`, `step_size`, and `learning_rate`. The lake-monster problem exhibits a _monotonicity_ property which may be leveraged in training. If an agent can succeed against a monster with a high speed, by taking the same path, it can succeed against a monster with a lower speed. This monotonicity may also hold with the `step_size` parameter: if an agent can escape from a monster when taking large steps, it can follow the same successful path by taking many smaller steps.

When training an agent, we often use a _mastery-based_ approach. In such an approach, an agent initially encounters a slow monster. Once the agent has shown mastery in this easier environment, the speed of the monster is increased. In this way, the agent is always paired with a monster whose speed perfectly matches the ability of the agent.

Extending this idea, the parameters `use_step_schedule` and `use_mastery` modify parameters over the course of training. Utilizing a mastery-based approach often leads to better learning outcomes. See [Results](#results) for a discussion.

Agent policies are periodically saved (see [Usage](#usage)) for later use. A well-trained policy can be revived and evaluated with modified parameters. The agents who received extensive training can often succeed when paired with high speed monsters simply by pushing environment parameters in the direction of continuous motion.

|                                                                                                              ![many agents](assets/many.gif)                                                                                                               |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| _A collection of agents in various states of training. Here, the monster is fixed in place and all agents are rotated toward the monster. Agents that stick to the shore have successfully escaped; those that wobble or wander will eventually be eaten._ |

### Nontrival Learning

Suppose that the agent runs antipodally away from the monster. The monster traverses the lake circumference, aiming for the point at which the agent will intersect with the shoreline. (The monster could make its semi-circumnavigation in a clockwise or counterclockwise motion; both paths lead to identical outcomes.) In such an episode, the agent travels a total distance of 1 whereas the monster travels a total distance of pi. Therefore, the agent succeeds with this strategy if and only if the speed of the monster is less than pi.

Because this strategy is so simple, it can be considered a baseline minimum for what the agent should aspire to achieve. In other words, the agent has learned something nontrivial if it can escape from a monster who has a speed of at least pi. Due to the discrete nature of the environment (specifically, the discrete variables `step_size` and `num_actions`), the agent would not be able to enact this exact strategy. Nevertheless, we still consider a monster speed of pi as a prerequisite for an intelligent agent.

With no knowledge of the mathematics, it is possible for a human to quickly learn to escape from a monster with speed pi. For example, the function `test_movement` in [`test_environment.py`](test_environment.py) uses simple handcrafted actions to succeed against a monster with speed 3.5. In the lake-monster problem, most humans could eventually fine-tune a set of actions to succeed against a monster with speeds between pi and 4.0. An agent can be described as having _human-level intelligence_ if it can succeed against monsters with speeds up to 4.0. An agent that can escape from a monster with speeds above 4.0 is considered to have _superintelligence_.

The [optimal mathematical solution](http://datagenetics.com/blog/october12013/index.html) allows an agent to escape a monster with speed 4.6, and so no agent will actually perform better than a handcrafted agent created by a human expert (a mathematician).

|           ![strong policy](assets/strong.gif)            |
| :------------------------------------------------------: |
| _An agent demonstrating highly complex learned actions._ |

## Results

In progress ....

## License

[MIT License](LICENSE.md)
