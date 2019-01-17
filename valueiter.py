import gym
import time
import numpy as np
#import pickle
env = gym.make('GridWorld-v0')
s = env.reset()
print(s)

#https://www.kaggle.com/charel/learn-by-example-reinforcement-learning-with-gym
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
#Q = np.zeros([NUM_STATES, NUM_ACTIONS]) 
V = np.zeros([NUM_STATES]) 
Pi = np.zeros([NUM_STATES], dtype=int)
gamma = 0.9 # discount factor
alpha = 0.1 # learning rate
epsilon = 0.1 #
significant_improvement = 0.01

def best_action_value(s):
    # finds the highest value action (max_a) in state s
    best_a = None
    best_value = float('-inf')

    # loop through all possible actions to find the best current action
    for a in range (NUM_ACTIONS):
        env.env.state = s
        env.env.statex = s % env.env.sizex
        env.env.statey = s // env.env.sizex
        s_new, rew, done, info = env.step(a) #take the action
        v = rew + gamma * V[s_new]
        #if s == 19 or s == 24:
        #    print("s, a, s_new, rew, done, v", s, a, s_new, rew, done, v)
        if v > best_value:
            best_value = v
            best_a = a
        #if s == 19 or s == 24:
        #    print("best_value, best_a", best_value, best_a)
    return best_a

iteration = 0
while True:
    # biggest_change is referred to by the mathematical symbol delta in equations
    biggest_change = 0
    for s in range (0, NUM_STATES):
        old_v = V[s]
        action = best_action_value(s) #choosing an action with the highest future reward
        env.env.state = s # goto the state
        env.env.statex = s % env.env.sizex
        env.env.statey = s // env.env.sizex

        s_new, rew, done, info = env.step(action) #take the action
        V[s] = rew + gamma * V[s_new] #Update Value for the state using Bellman equation
        #if s == 19 or s == 24:
        #    print('s, action, s_new, rew, done, V[s]', s, action, s_new, rew, done, V[s])
        Pi[s] = action
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    iteration += 1
    print(Pi)
    print(V)
    if biggest_change < significant_improvement:
        print (iteration,' iterations done')
        break
print(Pi)

s = env.reset()
env.render()
t = False
while not t:
    #s, r, t, c = env.step(int(random.random() * 4 + 1))
    #a = np.argmax(Q[s]) #env.action_space.sample()
    a = Pi[s]
    print(a)
    s, r, t, c = env.step(a) 
    print(s, r, t, c)
    env.render()
    time.sleep(0.5)
    if t:
        s = env.reset()
        env.render()
        t = False
        time.sleep(0.5)
time.sleep(100)
env.close()

