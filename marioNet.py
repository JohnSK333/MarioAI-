from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
#k = 5000
#i = 0
#tmp = []
#while(i < k):
#    tmp.append(-3)
#    i += 1
#np.save('actionList.npy',tmp)
rewards = []
newRewards = []
actionList = []
newAction = []
for i in np.load('rewards.npy'):
    rewards.append(i)
for i in np.load('actionList.npy'):
    actionList.append(i)

#for i in rewards:
#    print(i)
done = True
for step in range(5000):

    if done:
        state = env.reset()
    nextAction = env.action_space.sample()
    #print(nextAction)
    state, reward, done, info = env.step(nextAction)

    #print() 
    #print(rewards[step])
    if(reward < rewards[step]):
        nextAction = actionList[step]
        reward = rewards[step]
    env.render()
    newRewards.append(reward)
    newAction.append(nextAction)

np.save('rewards.npy',newRewards)
np.save('actionList.npy',newAction)
env.close()
