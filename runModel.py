from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np
import itertools
import sys
import matplotlib
import matplotlib.style
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from collections import defaultdict

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

def runModel():
    #load from file
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
        #Our key is a tuple of our [x,y] position
        #Each frame has a list of weights for each action
        Q[tuple(key)] = weights
    f.close()
    
    # Reset the environment and pick the first action
    state = env.reset()
    frame = tuple([40,0])
    prevFrame = tuple([39,0])
    nextFrame = tuple([0,0])
    prevAction = 0

    for t in itertools.count():
        bestAction = np.argmax(Q[frame])
        #long-jump(second half below)
        if(prevAction == 4):
            action = 4
        else:
            action = bestAction

        #take action
        next_state, reward, done, info = env.step(action)
        
        nextFrame = tuple([info['x_pos'],info['y_pos']])

        #We shouldn't die but just in case
        if done:
            break

        #Render Enviornment
        env.render()

        #Update for next frame
        state = next_state
        prevFrame = frame
        frame = nextFrame

        #The second half of our long-jump(simulates holding down jump)
        if((action ==  4) & (prevFrame[0] == frame[0])&(prevFrame[1] != frame[1])):
            prevAction = 4
        elif((action == 4) & (prevFrame[1] == frame[1])):
            prevAction = bestAction
        else:
            prevAction = action

runModel()
