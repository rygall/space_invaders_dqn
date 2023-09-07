import gymnasium as gym
import dqn
import numpy as np
import random


# instantiate dqn
agent = dqn.DQN(epsilon=0.75)

# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")

# define training parameters
max_epochs = 10000
max_episodes = 1
observation, info = env.reset()

# initialize actions 
actions = [0, 0, 0, 0, 0, 0]

# trackers
target_update_freq = 150

for episode in range(max_episodes):

    # print episode and reset environment
    print("Episode", episode)
    observation, info = env.reset()

    for epoch in range(max_epochs):

        # print epoch every 10 episodes
        if epoch % 10 == 0:
            print("Epoch:", epoch, "Actions:", actions)

        # get through first 33 frames since nothing happens
        if epoch < 100:
            observation, reward, terminated, truncated, info = env.step(np.random.randint(0, high=5, dtype=int))
            continue
        
        # get next action from DQN
        action = agent.action(observation)
        actions[action] += 1

        print(action)

        # take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # increment reward by 1 every epoch
        # encourages staying alive
        reward += 1

        # copy network to target network
        if (epoch % target_update_freq) == 0:
            target_update_freq = 50
            agent.updateTarget()
        
        if terminated or truncated:
            print("terminated/truncated")            
            observation, info = env.reset()
            break
        
        # train the DQN given new data
        agent.train(observation, reward)


    # save agent weights and actions
    np.save("episodeActions/actions_" + str(episode) + ".npy", actions)
    
    actions = np.zeros(max_epochs)
    
    agent.save(episode)

# close the environment  
env.close()