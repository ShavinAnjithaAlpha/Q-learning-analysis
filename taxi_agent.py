import numpy as np
import gym
import random
import pygame



# create a Taxi environment
env = gym.make('Taxi-v3', render_mode='human')

# create a new instaance of the taxi, and and get the initial state 
state = env.reset()

num_steps = 99
for s in range(num_steps + 1):
    print(f"step: {s} out of {num_steps}")

    # sample a random action from the list of avaialable actions 
    action = env.action_space.sample()
    
    # perform this action on the instance
    env.step(action)
    
    # print the new state
    env.render()
    
# end this instance of the taxi environment
env.close()