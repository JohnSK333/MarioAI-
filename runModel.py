from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np
import itertools
import sys
import matplotlib
import matplotlib.style
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from collections import defaultdict

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

#Standard Greedy Epsilon Policy
def greedyEpsilon(Q, epsilon, num_actions):
    def policyFunction(state,epoch,num_epochs):
        epsilon = 0.075 - (0.005*(epoch/num_epochs))
        actionChance = np.ones(num_actions,dtype = float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        actionChance[best_action] += (1.0 - epsilon) #Probability of a random action
        return actionChance
    return policyFunction


def qLearning(env, num_epochs, discount_factor = 1.0):
    #Epsilon is chance for random actions   
    epsilon = 0.1

    #load from file(pickle struggled with the weights)
    #so now we have this
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
            weights = np.append(weights,round(float(seperate[p]),4))
            p += 1
        #Our key is a tuple of our [x,y] position
        #Each frame has a list of weights for each action
        Q[tuple(key)] = weights
    f.close()
    #Brilliant File Management

    # Create a greedy algorithm
    policy = greedyEpsilon(Q, epsilon, env.action_space.n)

    with open('winSet.txt') as w:
        bestTime = int((w.read().split(','))[0])

    # Number of epochs
    for epoch in range(num_epochs):
        print("Epoch:  ",epoch)

        #Alpha is the learning rate(reduced over time)
        alpha = .3 - (.01 * (epoch/num_epochs))

        # Reset the environment and pick the first action
        state = env.reset()
        frame = tuple([40,0])
        prevFrame = tuple([38,0])
        nextFrame = tuple([0,0])
        prevAction = 0
        forceAction = False
        setThree = False
        counter = 0
        counter2 = 0
        actionList = [0]
        for t in itertools.count():
            # get probabilities of all actions for this frame
            action_chance = policy(frame,epoch,num_epochs)

            #generate random action
            #Ignore a few actions for training speed
            while(True):
                randAction = np.random.choice(
                        np.arange(len(action_chance)),
                        p = action_chance)
                if((randAction != 10) & (randAction != 11) & (randAction != 9) &
                        (randAction != 8) & (randAction != 7) & (randAction != 0)):
                    break

            #button press/release(second half below)
            if(prevAction == 4):
                action = 4  #Hold Button
            else:
                action = randAction

            if(setThree == True):
                action = 3
                counter2 = 0
                setThree = False
                #print(action)

            if(forceAction == True):
                action = 1
                forceAction = False

            #take action
            next_state, reward, done, info = env.step(action)
            actionList.append(action)

            #Record moveset of best time
            if((info['flag_get']) & (bestTime < info['time'])):
                bestTime = info['time']
                f = open('winSet.txt','w')
                f.write(str(bestTime)+',')
                for a in actionList:
                    f.write(str(a)+',')
                f.close()

            #The position we are moving to
            x = int(info['x_pos'])
            y = int(info['y_pos'])
            #Elimate  pixels(reduce file size)
            x = x - (x%3)
            y = y - (y%3)

            nextFrame = tuple([x,y])

            #Small penalty for making no progress
            if(prevFrame[0] >= frame[0]):
               reward -= .1

            #Temporal Difference Update(Reward Feedback)
            best_next_action = np.argmax(Q[nextFrame])
            td_target = reward + discount_factor * Q[nextFrame][best_next_action]
            td_delta = td_target - Q[frame][action]
            Q[frame][action] += alpha * td_delta

            if(reward < 4):
                Q[prevFrame][prevAction] += (alpha * td_delta)/2
            #Increasing the reward for moving quickly(Two frames updated)
            if((prevFrame[0]+5 < frame[0])  & (reward > 0)):
                Q[prevFrame][prevAction] += 0.1*(frame[0] - prevFrame[0]) * (alpha*td_delta)
                Q[frame][action] += 0.1*(frame[0] - prevFrame[0]) * (alpha*td_delta)

            #Rewards consecutive actions that improve our x position
            #This helps with jumping over tall pipes
            if((prevAction == action) & (reward > 1)):
                Q[frame][action] += .01 * alpha*td_delta
                Q[prevFrame][prevAction] += .01 * alpha*td_delta

            #done is True if epoch is terminated(out of lives)
            if done:
                break
            #Render Enviornment
            env.render()

            #The second half of our button press/release
            if(counter2 > 35):
                setThree = True
            if(action == 4):
                counter2 += 1
            if((action ==  4) & (prevFrame[0] == frame[0])&(prevFrame[1] != frame[1])):
                Q[frame][action] += .2
                prevAction = 4
                counter2 = 0
            elif((action == 4) & (prevFrame[0] < frame[0])):
                prevAction = randAction #prevent looping action 4
            else:
                prevAction = action

            if((frame[0] == prevFrame[0])&(frame[1] == prevFrame[1])):
                counter += 1
            else:
                counter = 0

            if((prevAction == 4) & (counter > 3)):
                forceAction = True
                counter = 0

            #Update for next frame
            state = next_state
            prevFrame = frame
            frame = nextFrame

    return Q

#Run program, second argument is number of epochs
Q = qLearning(env, 500)


