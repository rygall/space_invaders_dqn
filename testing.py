import gymnasium as gym
import dqn
import numpy as np

# instantiate dqn
agent = dqn.DQN(epsilon=1)

# file name to load weights from
load_weights_episode = "2"
agent.load(load_weights_episode)

print("Running with weights")

agent.print()

# instantiate environment
env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale")

observation, info = env.reset()
env.seed = 5

# define training parameters
max_epochs = 1000000
actions = np.zeros(max_epochs)

# run one episode
for epoch in range(max_epochs):   

    if epoch % 100 == 0:
        print("Epoch", epoch) 
    
    # get next action from DQN
    action = agent.action(observation)
    actions[epoch] = action
    print(action)
    
    # take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # copy network to target network
    if (epoch % 50) == 0:
        agent.updateTarget()
        
    # break if the environment terminates or truncates
    if terminated or truncated:
        observation, info = env.reset()
        break

# save agent weights and actions
np.save("actions_11.npy", actions)
env.close()

print("Running with actions")
env = gym.make("ALE/SpaceInvaders-v5", render_mode = "human", obs_type="grayscale")

observation, info = env.reset()
env.seed = 5

for epoch in range(max_epochs):    
    # take a step in the environment
    observation, reward, terminated, truncated, info = env.step(int(actions[epoch]))

    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()