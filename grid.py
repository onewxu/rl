import gym
import time
import numpy as np
import pickle
env = gym.make('GridWorld-v0')
s = env.reset()
print(s)

#https://www.kaggle.com/charel/learn-by-example-reinforcement-learning-with-gym
#https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
Q = np.zeros([NUM_STATES, NUM_ACTIONS]) 
gamma = 0.9 # discount factor
alpha = 0.2 # learning rate
epsilon = 0.1 #
print(Q)

for episode in range(1,1001):
    done = False
    rew_tot = 0
    obs = env.reset()
    while done != True:
            #action = np.argmax(Q[obs]) #choosing the action with the highest Q value 
            if np.random.rand(1) < epsilon:
                # exploration with a new option with probability epsilon, the epsilon greedy approach
                action = env.action_space.sample()
            else:
                # exploitation
                action = np.argmax(Q[obs])
            #print(action)
            obs2, rew, done, info = env.step(action) #take the action
            #print(obs2, rew, done, info)
            Q[obs,action] += alpha * (rew + gamma * np.max(Q[obs2]) - Q[obs,action]) #Update Q-marix using Bellman equation
            #Q[obs,action] = rew + gamma * np.max(Q[obs2]) # same equation but with learning rate = 1 returns the basic Bellman equation
            rew_tot = rew_tot + rew
            obs = obs2   
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode,rew_tot))
print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)

s = env.reset()
env.render()
t = False
while not t:
    #s, r, t, c = env.step(int(random.random() * 4 + 1))
    a = np.argmax(Q[s]) #env.action_space.sample()
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

