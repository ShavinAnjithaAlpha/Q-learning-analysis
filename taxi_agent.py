import csv
import numpy as np
import gym
import random
import pygame
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import time

PLOT = False
PLAY = False
DEBUG = True

def calculate_convergence_speed(convergence_values):
    
    convergence_criteria = 0.01
    
    # calculate the difference between each consecative values
    differences = np.diff(convergence_values)
    
    # find the first value where the values are less than the convergence criteria 
    converged_index = np.argmax(convergence_values < convergence_criteria)
    
    if converged_index == 0:
        # did not converge within the number of episodes
        return -1
    
    # fit a linear regression model to estimate the convergence speed
    convergence_speed, _, _, _, _ = linregress(np.arange(len(differences[:converged_index])), differences[:converged_index])

    if converged_index > 0:
        # print(f"Converged after {converged_index} iterations")
        return convergence_speed
    else:
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
    
    return convergence_values

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

def get_mean_convergence_speed(state_size, action_size, env, learning_rate, discount_rate):
    
    train_rounds = 20
    
    convergence_speed_data = np.zeros(train_rounds)
    
    # start time of the calculation
    if DEBUG: start_time = time.time()
    for  i in range(train_rounds):
        
        # initialize the q table
        qtable = np.zeros((state_size, action_size))
        # train the agent for a given learning rate and discount rate, plot the convergence graph of the qtable
        convergence_values = train(qtable, env, learning_rate, discount_rate)
        # print the convergence speed
        convergence_speed = calculate_convergence_speed(convergence_values)
        if (convergence_speed == -1):
            if DEBUG: print("Did not converge")
            return -1
        
        convergence_speed_data[i] = convergence_speed
        
        if PLAY:
            # play the trained agent
            play(qtable, env)
    
    # end time of the calculation
    if DEBUG: end_time = time.time()
    
    if DEBUG:
        print(f"elapsed time: {end_time - start_time:.4f}")
       
    speed_mean = np.mean(convergence_speed_data)
    speed_std = np.std(convergence_speed_data)
    if DEBUG: print(f"Convergence speed: {speed_mean:.5f}")
    return speed_mean, speed_std   

def generate_heat_map(state_size, action_size, env):
    
    file = open("convergence_data.csv", "w", newline='')
    writer = csv.writer(file)
    writer.writerow(['Learning Rate', 'Discount Factor', 'Convergence Speed'])
    # numy array to hold the convergence speed data 10x10
    convergence_speed_data = np.zeros((11, 11))
    
    for learning_rate in range(11):
        for discount_factor in range(11):
            if DEBUG: print(f"(x, y) = ({learning_rate}, {discount_factor})")
            # calculate the convergence speed for each learning rate and discount factor
            ret = get_mean_convergence_speed(state_size, action_size, env, learning_rate/10, discount_factor/10)
            if ret == -1:
                convergence_speed_data[learning_rate, discount_factor] = 0
                writer.writerow([learning_rate/10, discount_factor/10, 0])
            else:
                convergence_speed_data[learning_rate, discount_factor] = ret[0] * 100 # increase the scale of the heatmap
                writer.writerow([learning_rate/10, discount_factor/10, ret[0]])
    
    
    file.close()
    # plot the heatmap
    sns.heatmap(convergence_speed_data, annot=True)
    
    # add labels to the plot
    plt.xlabel("Discount Factor")
    plt.ylabel("Learning Rate")
    plt.title("Convergence Speed")
    
    # show the plot
    plt.show()

def generate_convergence_data(state_size, action_size, env):
    
    discount_rate = 0.8
    
    # write the convergence data to a csv file
    with open('convergence_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Learning Rate', 'Convergence Speed', 'Standard Deviation'])
        for learning_rate in range(11):
            learning_rate /= 10
            ret = get_mean_convergence_speed(state_size, action_size, env, learning_rate, discount_rate)
            if ret == -1:
                continue
            mean_convergence_speed, std_convergence_speed = ret
            writer.writerow([learning_rate, mean_convergence_speed, std_convergence_speed])
            

def main():
    # create a Taxi environment
    env = gym.make('Taxi-v3', render_mode='rgb_array')
    
    # qtable dimensions
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    # generate the convergence data
    # generate_convergence_data(state_size, action_size, env)
    generate_heat_map(state_size, action_size, env)
            
    env.close()
        
        
if __name__ == "__main__":
    main()