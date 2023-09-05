import gymnasium as gym
import dqn
import numpy as np


# instantiate dqn
agent = dqn.DQN()

# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")

# define training parameters
max_epochs = 34
max_episodes = 100
observation, info = env.reset()

episode_actions = np.zeros((max_episodes, max_epochs))
curr_actions = np.zeros(max_epochs)

for episode in range(max_episodes):
    print("Episode", episode)
    observation, info = env.reset()

    for epoch in range(max_epochs):

        # get through first 33 frames since nothing happens
        if epoch < 33:
            observation, reward, terminated, truncated, info = env.step(1)
            continue
        
        # get next action from DQN
        action = agent.action(observation)
        curr_actions[epoch] = action

        # take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # train the DQN given new data
        agent.train(observation, reward, epoch)

        # copy network to target network
        if (epoch % 50) == 0:
            agent.updateTarget()
        
        if terminated or truncated:
            observation, info = env.reset()
            break

    episode_actions[episode] = curr_actions
    curr_actions = np.zeros(max_epochs)

        # save agent weights and actions
    np.save("episodeActions/Episode-Actions_" + str(episode) + ".npy", episode_actions)
    agent.save(episode)

# close the environment  
env.close()