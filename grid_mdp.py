import logging
import numpy
import random
from gym import spaces
import gym
from gym.envs.classic_control import rendering

logger = logging.getLogger(__name__)

class GridEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.observation_space = spaces.Discrete(25) #state space
        self.states = [0,1,5,6,7,8,9,12,13,14,15,16,17,19,20,21,22,24];
        self.state = None
        self.statex = None
        self.statey = None

        self.obstacles = [2,3,4,10,11,18,23] #obstacle states
        self.obstacle_states = dict()  
        self.obstacle_states[2] = 1
        self.obstacle_states[3] = 1
        self.obstacle_states[4] = 1
        self.obstacle_states[10] = 1
        self.obstacle_states[11] = 1
        self.obstacle_states[18] = 1
        self.obstacle_states[23] = 1

        self.terminate = 14;  #terminate state
        self.terminate_states = dict()  
        self.terminate_states[14] = 1

        self.sizex = 5; 
        self.sizey = 5;

        self.action_space = spaces.Discrete(4)

        #self.seed()
        self.gamma = 0.8         #discount factor
        self.viewer = None

    def step(self, action):
        #system current state
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}

        statex = self.statex
        statey = self.statey
        if action == 0 and statex + 1 < self.sizex:
            statex = statex + 1
        if action == 1 and statex > 0:
            statex = statex - 1
        if action == 2 and statey + 1 < self.sizey:
            statey = statey + 1
        if action == 3 and statey > 0:
            statey = statey - 1
        next_state = statey * self.sizex + statex
        is_terminal = False
        r = 0
        if next_state in self.obstacle_states:
            return state, r, is_terminal, {}

        if next_state in self.terminate_states:
            is_terminal = True
            r = 1 
        self.state = next_state
        self.statex = statex
        self.statey = statey
        return next_state, r,is_terminal,{}
    def reset(self):
        self.state = self.states[int(random.random() * len(self.states))]
        self.statex = self.state % self.sizex
        self.statey = self.state // self.sizey
        return self.state
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 600
        screen_height = 600
        rwidth = 80
        l,r,t,b = 100,100+rwidth,100+rwidth,100

        if self.viewer is None:
            #from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #create grid world
            for index in range(0, 6):
                line1 = rendering.Line((l,b+index*rwidth), (l+self.sizex*rwidth,b+index*rwidth))
                line1.set_color(0.5, 0, 0)
                self.viewer.add_geom(line1)
                line2 = rendering.Line((l+index*rwidth,b), (l+index*rwidth,b+self.sizey*rwidth))
                line2.set_color(0.5, 0, 0)
                self.viewer.add_geom(line2)

            #create obstacles
            for index in range(0, len(self.obstacles)):
                pole1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                pole1.set_color(0,0,0)
                tx, ty = self.obstacles[index] % self.sizex, self.obstacles[index] // self.sizex
                self.poletrans1 = rendering.Transform(translation=(rwidth*tx, rwidth*ty))
                pole1.add_attr(self.poletrans1)
                self.viewer.add_geom(pole1)

            #create terminal
            pole2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole2.set_color(0.2,0.8,0.4)
            tx, ty = self.terminate % self.sizex, self.terminate // self.sizex
            self.poletrans2 = rendering.Transform(translation=(rwidth*tx, rwidth*ty))
            pole2.add_attr(self.poletrans2)
            self.viewer.add_geom(pole2)

            #create robot
            tx, ty = self.statex + 0.5, self.statey + 0.5
            self.robot= rendering.make_circle(30) 
            self.robotrans = rendering.Transform(translation=(l+rwidth*tx, b+rwidth*ty))
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.2, 0.2)
            self.viewer.add_geom(self.robot)

        if self.state is None: return None
        tx, ty = self.statex + 0.5, self.statey + 0.5
        self.robotrans.set_translation(l+rwidth*tx, b+rwidth*ty) #translate robot

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

