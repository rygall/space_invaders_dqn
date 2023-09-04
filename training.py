import gymnasium as gym
import dqn
import numpy as np


# instantiate dqn
agent = dqn.DQN()

# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", obs_type="grayscale")

# define training parameters
max_epochs = 1000
observation, info = env.reset()
    
for epoch in range(1, max_epochs):
    
    # get next action from DQN
    action = agent.getAction(observation)

    # take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # train the DQN given new data
    agent.train(action, observation, reward)
        
    if terminated or truncated:
        observation, info = env.reset()
        
env.close()