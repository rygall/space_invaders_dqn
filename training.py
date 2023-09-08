import gymnasium as gym
import dqn
import numpy as np
import random


# instantiate dqn
agent = dqn.DQN(epsilon=0.25)

# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")

# define training parameters
max_epochs = 10000
max_episodes = 4
observation, info = env.reset()

# initialize actions 
actions = [0, 0, 0, 0, 0, 0]

# param
target_update_freq = 50

# trackers
episode_reward = 0
rewards = []
final_epoch = 0
total_epochs = []

for episode in range(3, max_episodes):

    # print episode and reset environment
    print("Episode", episode)
    observation, info = env.reset()
    actions = [0, 0, 0, 0, 0, 0]

    for epoch in range(max_epochs):

        # print epoch every 10 episodes
        if epoch % 10 == 0:
            print("Epoch:", epoch, "Actions:", actions)

        # get through first 33 frames since nothing happens
        if epoch < 33:
            observation, reward, terminated, truncated, info = env.step(np.random.randint(0, high=5, dtype=int))
            continue
        
        # get next action from DQN
        action = agent.action(observation)
        actions[action] += 1
        print(action)

        # take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # copy network to target network
        if (epoch % target_update_freq) == 0:
            agent.updateTarget()
        
        if terminated or truncated:
            print("terminated/truncated")            
            observation, info = env.reset()
            break

        if epoch % 100 == 0:
            agent.print()
        
        # train the DQN given new data
        agent.train(observation, reward)

        episode_reward += reward
        final_epoch = epoch

    # save agent weights
    agent.save(episode)

    # update agents gamma
    agent.setEpsilon(0.75)

    #store total rewards and total epochs
    rewards.append(episode_reward)
    total_epochs.append(final_epoch)


print("Episode Rewards:", rewards)
print("Episode Epoch:", total_epochs)


# close the environment  
env.close()