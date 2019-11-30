from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np
import itertools
import sys
import matplotlib
import matplotlib.style
import pandas as pd
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from collections import defaultdict
import plotting
import re
matplotlib.style.use('ggplot')

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

def greedyEpsilon(Q, epsilon, num_actions):
    def policyFunction(state):
        Action_probabilities = np.ones(num_actions,dtype = float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities
    return policyFunction 

def qLearning(env, num_episodes, discount_factor = 1.0, alpha = .6, epsilon = 0.05):
    # state -> (action -> action-value)
            
    #loadQ = np.load('Brain3.npy',allow_pickle = True)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    f = open('Brain.txt','r')
    J = defaultdict()
    l = []
    seperate = []
    key = []
    combine = []
    for line in f:
        l.append(line)
    for i in l:
        weights = np.array([])
        i = i.replace('(','').replace(')','').replace(' ','')
        seperate = i.split(',')
        key = [int(seperate[0]),int(seperate[1])]
        p = 2
        while(p < 14):
            weights = np.append(weights,float(seperate[p]))
            p += 1

        Q[tuple(key)] = weights
        
    f.close()
    
    for p in list(Q.items()):
        print(p,"\n")

    # Keeps track of useful statistics 
    stats = plotting.EpisodeStats( 
            episode_lengths = np.zeros(num_episodes), 
	    episode_rewards = np.zeros(num_episodes))
    
    # Create an epsilon greedy policy
    policy = greedyEpsilon(Q, epsilon, env.action_space.n)

    # For every episode 
    for epoch in range(num_episodes):
        print("Epoch:  ",epoch)
        # Reset the environment and pick the first action
        state = env.reset()
        frame = tuple([40,0])
        prevFrame = tuple([39,0])
        nextFrame = tuple([0,0])
        prevAction = 0

        for t in itertools.count():
            # get probabilities of all actions from current state
            action_probabilities = policy(frame)
            # choose action according to
            # the probability distribution
            action = np.random.choice(
                    np.arange(len(action_probabilities)),
                    p = action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done, info = env.step(action)
            
            nextFrame = tuple([info['x_pos'],info['y_pos']])
            #print(nextFrame)
            # Update statistics
            stats.episode_rewards[epoch] += reward
            stats.episode_lengths[epoch] = t
            
            if(action == 4):
                reward += .002
            if(action == 3):
                reward += .001

            if((prevFrame[0] == frame[0]) & (prevFrame[1] == frame[1])):
               reward -= .01
            # TD Update
            best_next_action = np.argmax(Q[nextFrame])
            td_target = reward + discount_factor * Q[nextFrame][best_next_action]
            td_delta = td_target - Q[frame][action]
            Q[frame][action] += alpha * td_delta
            
            #if((prevFrame[0]+5 < (frame[0])) & (prevFrame[1] < frame[1]) & (reward > 0)):
                #print("Frame: ",frame[0], " ",frame[1])
                #print("Previous Frame: ",prevFrame[0], " ",prevFrame[1])
                #Q[prevFrame][prevAction] += 2*(alpha*td_delta)
                #Q[frame][action] += alpha * td_delta
            
            #if((prevAction == action) & (reward > 1)):
                #print("Reward: ",reward," Action: ", action)
                #Q[frame][action] += alpha*td_delta
            
            # done is True if episode terminated
            if done:
                break
            env.render()
            state = next_state
            prevFrame = frame
            frame = nextFrame
            prevAction = action
        f = open('Brain.txt','w')
        for i in Q:
            f.write(str(i))
            for j in Q[i]:
                f.write(','+str(j))
            f.write('\n')
        f.close()

    return Q, stats

Q, stats = qLearning(env, 20) 

f = open('Brain.txt','w')
for i in Q:
    f.write(str(i))
    for j in Q[i]:
        f.write(','+str(j))
f.write('\n')
f.close()

