import numpy as np
import gym
import random
import pygame



def main():
    # create a Taxi environment
    env = gym.make('Taxi-v3', render_mode='rgb_array')
    
    # initialize the q table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))
    
    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005
    
    # training variables
    num_episodes = 1000
    max_steps = 99 # per spisode
    
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
        
    env.close()
        
        
if __name__ == "__main__":
    main()