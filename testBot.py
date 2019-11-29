from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

actionList = []

f = open('actionList.txt','r')
newFile = f.read()
data = newFile.splitlines()
f.close()
for i in data:
    actionList.append(int(i))

done = True
for i in actionList:
    if done:
        state = env.reset()
    state, reward, done, info = env.step(i)

    env.render()

env.close()

