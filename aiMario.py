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
        epsilon = 0.075 - (0.075*(epoch/num_epochs))
        actionChance = np.ones(num_actions,dtype = float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        actionChance[best_action] += (1.0 - epsilon) #Probability of a random action
        return actionChance
    return policyFunction 

def qLearning(env, num_epochs, discount_factor = 1.0):
    #Epsilon is chance for random actions   
    epsilon = 0.2
    
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
            weights = np.append(weights,float(seperate[p]))
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

    print(bestTime)

    # Number of epochs
    for epoch in range(num_epochs):
        print("Epoch:  ",epoch)
        
        #Alpha is the learning rate(reduced over time)
        alpha = .8 - (.8 * (epoch/num_epochs))

        # Reset the environment and pick the first action
        state = env.reset()
        frame = tuple([40,0])
        prevFrame = tuple([38,0])
        nextFrame = tuple([0,0])
        prevAction = 0
        setRelease = False
        actionList = [0]
        for t in itertools.count():
            # get probabilities of all actions for this frame
            action_chance = policy(frame,epoch,num_epochs)

            #generate random action
            #He keeps going down the pipe and it is a four minute wait to learn not to
            while(True):
                randAction = np.random.choice(
                        np.arange(len(action_chance)),
                        p = action_chance)
                if(randAction != 10):
                    break

            #button press/release(second half below)
            if(setRelease & (prevAction == 4)):
                action = 3  #Button Release
            elif(prevAction == 4):
                action = 4  #Hold Button
            else:
                action = randAction
            
            #take action
            next_state, reward, done, info = env.step(action)
            actionList.append(action)

            if((info['flag_get']) & (bestTime < info['time'])):
                bestTime = info['time']
                f = open('winSet.txt','w')
                f.write(str(bestTime)+',')
                for a in actionList:
                    f.write(str(a)+',')
                f.close()
            #The position we are moving to
            x = info['x_pos']
            y = info['y_pos']
            #Elimate all odd pixels(reduce file size)
            x = x - (x%2)
            y = y - (y%2)

            nextFrame = tuple([x,y])
            
            #Small penalty for making no progress
            if((prevFrame[0] == frame[0]) & (action != 4)):
               reward -= .1
            
            #Temporal Difference Update(Reward Feedback)
            best_next_action = np.argmax(Q[nextFrame])
            td_target = reward + discount_factor * Q[nextFrame][best_next_action]
            td_delta = td_target - Q[frame][action]
            Q[frame][action] += alpha * td_delta
            
            #Increasing the reward for moving quickly(Two frames updated)
            if((prevFrame[0]+3 < frame[0])  & (reward > 0)):
                Q[prevFrame][prevAction] += 0.1*(frame[0] - prevFrame[0]) * (alpha*td_delta)
                Q[frame][action] += 0.1*(frame[0] - prevFrame[0]) * (alpha*td_delta)
            
            #Rewards consecutive actions that improve our x position
            #This helps with jumping over tall pipes
            if((prevAction == action) & (reward > 1)):
                Q[frame][action] += .1 * alpha*td_delta
                Q[prevFrame][prevAction] += .1 * alpha*td_delta

            #done is True if epoch is terminated(out of lives)
            if done:
                break

            #Render Enviornment
            env.render()

            #Update for next frame
            state = next_state
            prevFrame = frame
            frame = nextFrame

            #The second half of our button press/release
            if((action ==  4) & (prevFrame[0] == frame[0])&(prevFrame[1] != frame[1])):
                prevAction = 4 
            elif((action == 4) & (prevFrame[1] == frame[1])):
                prevAction = randAction
            else:
                prevAction = action

            if((prevFrame[0] < frame[0])&(prevFrame[1] == frame[1])):
                setRelease = True
            else:
                setRelease = False
        #Save our model every ten epochs
        sorted(Q.items(), key = lambda k_v: k_v[1][2], reverse = False)
        if((epoch%10) == 0):
            f = open('Brain.txt','w') 
            for i in Q:
                f.write(str(i))
                for j in Q[i]:
                    f.write(','+str(j))
                f.write('\n')
            f.close()

    return Q

#Run program, second argument is number of epochs
Q = qLearning(env, 300) 

#Final save once training is completed
f = open('Brain.txt','w')
for i in Q:
    f.write(str(i))
    for j in Q[i]:
        f.write(','+str(j))
f.write('\n')
f.close()

