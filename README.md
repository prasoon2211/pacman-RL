# Playing pacman using Reinforecement Learning

This code trains a reinforcement learning agent to play PacMan by using only the pixels on the screen.

This repository contains two models:
- A vanilla Deep Q-Network with experience replay
- An enhanced Deep Q-Network with experience replay, Double DQN weights and uses a Dueling architecture

The deep neural net is modeled in tensorflow and we use the Open AI Gym to generate the game simulation. We also use OpenAI baselines - which is a wrapper over tensorflow - for writing the model.

### Requirements:

- Python 3.5
- tensorflow=0.1.4
- Open AI Gym
- Open AI Gym[atari]
- numpy
- baselines=0.1.4
- scikit-learn


### Installation issues:
- Open AI baselines needs Mujoco 1.5 which is a proprietary software. The installation instructions provided on the baselines' GitHub repo mention this step as part of the installation process. You need to install Mujoco and enter the activation code that can be found from Mujoco's website. This enables a 30 day trial.
- If you are facing problems in the installation of Mujoco 1.5, a mujoco 0.5.7 installation can also work but it needs an older version of baselines (0.1.3). This is a breaking change and the code provided in the solution needs some small modifications to make it work.

### Code structure

The code has two major directories:
- `python_full_DQN`
- `python_vanilla_DQN`
- `training_logs`

Both directories contain a `pacman_agent.py` file which implements the RL agent, a `save_model` directory which contains a pre-trained model and a `logs` directory which contains execution logs in csv format.

The `training_logs` directory contains some renamed log files for earlier training sessions that have recoreded the step count, episode count and the episode rewars for different values of hyperparameters.

### Training

To train, go into any one of the `python_*_DQN` directories and open the file `pacman_agent.py`. On the top section of the file, you can set any hyperparameters of the model. If you want to continue training from the last saved checkpoint, set the flag `is_load_model=True` otherwise set it to `False`. Then, run:

```
$ python pacman_agent.py
```
This will commence training of the agent. If a saved model checkpoint is loaded, the training will continue from where it left off. Otherwise, the training will begin from scratch. Make sure to change `NUM_STEPS` variable to a large enough number so that the training doesn't stop before you want it to, especially if you're loading an old checkpoint.

(Note: It takes upto 15 hours of training for Full DQN and upto 6 hours of training for vanilla DQN to complete 2,000,000 time steps on a machine with an Nvidia 860m Grapics card, an 8GB RAM and a Core i7 CPU. The training can be continued for far longer than 2 million time steps for better results).

### Watching the agent play

In either of the `python_*_DQN` directories, open the `pacman_agent.py` file and the set the flag `watch_train=True` and `is_load_model=True`. This will load the pretrained model in memory and play the game based on the the Q-values predicted by the model. This will also keep training the model as the more games are played so that the model can improve in the background.

Both `python_*_DQN` directories contain a CSV file, called `play_500_eps.csv` which contains reward data from playing PacMan for 500 episodes. This data is then used to calculate metrics to compare the two models.

### Visualizing metrics

The progress made on training or playing the game can be visualized using the CSV log files. These files store three key data points: the number of time steps, the number of episodes passed and the reward per episode. The easiest visualization is to make line plot of number of steps vs the episode reward. This graph is usually very jittery therefore taking a moving average over a fixed number of steps can be helpful in visualizing long term trends.
























