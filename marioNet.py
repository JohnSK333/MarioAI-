from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
while(quit != 1):
    quit = 0
    rewards = []
    newRewards = []
    actionList = []
    newAction = []

    f = open('actionList.txt','r')
    newFile = f.read()
    data = newFile.splitlines()
    f.close()
    for k in data:
        actionList.append(int(k))
        
    done = True
    for step in range(5000):

        if done:
            state = env.reset()
        nextAction = env.action_space.sample()
        state, reward, done, info = env.step(nextAction)
        
        if(reward > 0):
            nextAction = actionList[step]
        
        env.render()

        newAction.append(nextAction)

    open('actionList.txt', 'w').close()
    f = open('actionList.txt','w')
    for j in newAction:
        f.write(str(j)+"\n")
    f.close()

    env.close()
    quit += 1
