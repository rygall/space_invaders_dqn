import gymnasium as gym
import dqn
import numpy as np

# instantiate dqn
agent = dqn.DQN()

# file name to load weights from
load_weights_episode = "6"
agent.load(load_weights_episode)

print("Running with weights")
# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")

observation, info = env.reset()
env.seed = 0

# define training parameters
max_epochs = 1000000
actions = np.zeros(max_epochs)

for epoch in range(max_epochs):   
    if epoch % 100 == 0:
        print("Epoch", epoch) 
    
    # get next action from DQN
    action = agent.action(observation)
    actions[epoch] = action
    print(action)
    
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

# save agent weights and actions
np.save("actions.npy", actions)
env.close()

print("Running with actions")
env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human", obs_type="grayscale")

observation, info = env.reset()
env.seed = 0

for epoch in range(max_epochs):    
    # take a step in the environment
    observation, reward, terminated, truncated, info = env.step(int(actions[epoch]))

    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()