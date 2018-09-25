import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
# from utils import *

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        # params
        self.m = 1
        self.l = 1
        self.g = 10
        self.b = 0.05
        self.max_speed=8
        self.max_torque=2.
        self.noise_thresh = 1e-1
        self.dt=.02
        self.viewer = None

        # TODO: don't think this is right
        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta
        th = self._unwrap_angle(th)
        g = self.g
        m = self.m
        l = self.l
        b = self.b
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)
        # add noise perturbation
        if np.abs(th - np.pi) < .005:
            if np.random.uniform(0,1) < self.noise_thresh:
                # u = np.random.choice([-20,20])
                u = np.random.uniform(-20,20)

        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (3./(m*l**2)*(u - b*thdot) - 3*g/(2*l) * np.sin(th)) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        # high = np.array([np.pi/4, np.pi/4])
        # self.state = self.np_random.uniform(low=-high, high=high)
        self.state = self.np_random.uniform(low=-.25, high=.25, size=(2,))
        # self.state[0] += np.pi
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        # TODO: figure out if its better to return cos/sin angle
        # return np.array([np.cos(theta), np.sin(theta), thetadot])
        theta = unwrap_angle(theta)
        return np.array([theta, thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.3, .8, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/counterclockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] - np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

    def _unwrap_angle(self, theta):
        sign = (theta >=0)*1 - (theta < 0)*1
        theta = np.abs(theta) % (2 * np.pi)
        return sign*theta

    def _linearize(self, theta):
        m = self.m
        g = self.g
        l = self.l
        b = self.b
        c = np.cos(theta)
        A = np.array([[0,1],[-c*g/l, -b/(m*l**2)]])
        B = np.array([[0], [1/(m*l**2)]])
        return A, B

    def total_energy(self):
        th, thdot = self.state
        m = self.m
        g = self.g
        l = self.l
        c = np.cos(th)
        return 0.5 * m * l**2 * thdot**2 - m*g*l*c

    def desired_energy(self):
        m = self.m
        g = self.g
        l = self.l
        return m*g*l

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def unwrap_angle(theta):
    sign = (theta >=0)*1 - (theta < 0)*1
    theta = np.abs(theta) % (2 * np.pi)
    return sign*theta
