from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np
import itertools
import sys
import pickle
import dill
import matplotlib
import matplotlib.style
import pandas as pd
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from collections import defaultdict
import plotting
matplotlib.style.use('ggplot')

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    def policyFunction(state):
        Action_probabilities = np.ones(num_actions,dtype = float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities
    return policyFunction 

def qLearning(env, num_episodes, discount_factor = 1.0, alpha = 0.6, epsilon = 0.1):
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value)
    loadQ = np.load('Brain.npy',allow_pickle = True)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    Q.update(loadQ.item())
    #print(list(Q.items()))
    # Keeps track of useful statistics 
    stats = plotting.EpisodeStats( 
            episode_lengths = np.zeros(num_episodes), 
	    episode_rewards = np.zeros(num_episodes))
    
    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)

    # For every episode 
    for ith_episode in range(num_episodes):
        print("Episode:  ",ith_episode)
        # Reset the environment and pick the first action
        state = env.reset()
        frame = 40;
        nextFrame = 0;
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
            
            nextFrame = info['x_pos']
            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t
            
            # TD Update
            best_next_action = np.argmax(Q[nextFrame])
            td_target = reward + discount_factor * Q[nextFrame][best_next_action]
            td_delta = td_target - Q[frame][action]
            Q[frame][action] += alpha * td_delta

            # done is True if episode terminated
            if done:
                break
            env.render()
            state = next_state
            frame = nextFrame
            np.save('Brain',np.array(dict(Q)))
    return Q, stats

Q, stats = qLearning(env, 100) 
np.save('Brain',np.array(dict(Q)))

