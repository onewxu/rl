import gym
import time
env = gym.make('GridWorld-v0')
env.reset()
t = False
while not t:
    #s, r, t, c = env.step(int(random.random() * 4 + 1))
    a = env.action_space.sample()
    #print(a)
    s, r, t, c = env.step(a)
    print(s, r, t, c)
    env.render()
    time.sleep(0.1)
time.sleep(1)
env.close()


