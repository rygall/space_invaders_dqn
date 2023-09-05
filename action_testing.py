import gymnasium as gym
import numpy as np


# file name to load actions from
load_action = "episodeActions/Episode-Actions_9.npy"

# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human", obs_type="grayscale")

# define training parameters
max_epochs = 1000
max_episodes = 100
observation, info = env.reset()

curr_actions = np.load(load_action)

for epoch in range(max_epochs):    
    # take a step in the environment
    print(curr_actions[20])
    observation, reward, terminated, truncated, info = env.step(int(curr_actions[20][epoch]))
    
    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()