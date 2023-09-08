The environment that we used to run the code requires Gymnasium to be installed through python. We also used the numba library to speed up execution of some of our functions. 

From an existing python installation, run:
pip install "gymnasium[all]"

This should install the gymnasium dependencies. To be able to use the Space Invaders environment, you also need to run:
pip install "gym[accept-rom-license, atari]"

Finally, in order to utilize the numba library, you need to run:
pip install numba

This should provide an environment to run the code. From there, run training.py to start training the agent. The values for episodes and epochs can be altered as desired.

After training is complete, either action-testing.py or testing.py can be used. 

action-testing.py will take in a load_action, which should be saved in the proper episodeActions folder after training, and a suffix for the episode you want to run. Then, it will run it through the gymnasium environment and play through the action sequence in real time from that episode.

testing.py tales om a load_weights_episode which is a single character that represents the episode to load the weights from. These weights should have been properly saved in the saves folder from training.py, and the weights will get loaded using the agent.load function. 
testing.py will then run that episode's weights through the environment to determine the action sequence to play, and then subsequently play that action sequence. The first part has render mode turned off, so the game window will only display once the action sequence is running.