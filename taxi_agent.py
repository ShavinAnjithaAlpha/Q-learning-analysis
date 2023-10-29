import numpy as np
import gym
import random
import pygame
import matplotlib.pyplot as plt
from scipy.stats import linregress

PLOT = False
PLAY = False

def exponential_decay(x, a, b):
    return a * np.exp(b * x)

def calculate_convergence_speed(convergence_values):
    
    convergence_criteria = 0.01
    
    # calculate the difference between each consecative values
    differences = np.diff(convergence_values)
    
    # find the first value where the values are less than the convergence criteria 
    converged_index = np.argmax(convergence_values < convergence_criteria)
    
    # fit a linear regression model to estimate the convergence speed
    convergence_speed, _, _, _, _ = linregress(np.arange(len(differences[:converged_index])), differences[:converged_index])

    if converged_index > 0:
        # print(f"Converged after {converged_index} iterations")
        return convergence_speed
    else:
        # print("Did not converge within the specified criterion.")
        return -1

def train(qtable, env, learning_rate, discount_rate):
    # create a numpy array to hold the qtable convergence values in each steps
    convergence_values = np.zeros(1000)
    
    # hyper parameters
    epsilon = 1.0
    decay_rate = 0.005
    
    # training variables
    num_episodes = 1000
    max_steps = 99 # per spisode
    
    previous_qtbale = np.copy(qtable)
    # training the agent
    for episode in range(num_episodes):
         
         # reset the environment
        state, _ = env.reset()
        done = False
         
        for s in range(max_steps):
             
             # explore or exploit tradeoff
            if random.uniform(0, 1) < epsilon:
                 # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :])
                
            # take action and observe reward
            new_state, reward, done, truncated, info  = env.step(action)
            
            # apply the Q-learing algorithm
            qtable[state, action] += learning_rate * (reward + discount_rate * np.max(qtable[new_state,:]) - qtable[state, action])
            
            # upload to our new state
            state = new_state
            
            # if done then finish episode
            if done == True:
                break
            
        # update epsilon
        epsilon *= np.exp(-decay_rate * episode)
        
        # Calculate the change between current and previous iterations
        change = np.linalg.norm(qtable - previous_qtbale)
        convergence_values[episode] = change
        previous_qtbale = np.copy(qtable)
    
    if PLOT:
        # plot the convergence values
        plt.plot(convergence_values)
        plt.xlabel("Episode")
        plt.ylabel("Convergence")
        plt.show()
    
    # print the convergence speed
    convergence_speed = calculate_convergence_speed(convergence_values)
    print(f"Convergence speed: {convergence_speed:.5f}")

def play(qtable, env):
    
    num_episodes = 1000
    max_steps = 99 # per episode
    
    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")
    
    
    # watch our agent play
    state, _ = env.reset()
    done = False
    rewards = 0
    
    for s in range(max_steps):
        
        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))
        
        action = np.argmax(qtable[state, :])
        new_state, reward, done, truncated, info = env.step(action)
        rewards += reward
        
        image = env.render()
        print(f"score: {rewards}")
        state = new_state
        
        if done:
            break

def main():
    # create a Taxi environment
    env = gym.make('Taxi-v3', render_mode='rgb_array')
    
    # initialize the q table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))
    
    # hyperparameters
    learning_rate = 0.7
    discount_rate = 0.8
    
    # train the agent for a given learning rate and discount rate, plot the convergence graph of the qtable
    train(qtable, env, learning_rate, discount_rate)
    
    if PLAY:
        # play the trained agent
        play(qtable, env)
    
        
    env.close()
        
        
if __name__ == "__main__":
    main()