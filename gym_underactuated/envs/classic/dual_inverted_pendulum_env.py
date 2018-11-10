"""
Modification of the Classic dual inverted pendulum system implemented by Kent H. Lundberg et al.
Code adopted from OpenAI Gym implementation
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class DualInvertedPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # gravity
        self.gravity = self.g = 9.8

        # cart
        self.m0 = self.M = self.masscart = 1.0

        # pole 1
        self.m1 = 0.1
        self.l1 = self.len1 = 1.0 # actually half the pole's length
        self.I1 = (1/12) * self.m1 * (2*self.len1)**2
        self.inertia_1= (self.I1 + self.m1 * self.len1**2)
        self.pm_len1 = self.m1 * self.len1

        # pole 2
        self.m2 = 0.05
        self.l2 = self.len2 = 0.5 # actually half the pole's length
        self.I2 = (1/12) * self.m2 * (2*self.len2)**2
        self.inertia_2 = (self.I2 + self.m2 * self.len2**2)
        self.pm_len2 = self.m2 * self.len2

        # Other params
        self.total_mass = (self.masscart + self.m1 + self.m2)
        self.force_mag = 10.0
        self.dt = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 200.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        c1 = math.cos(theta1)
        s1 = math.sin(theta1)
        c2 = math.cos(theta2)
        s2 = math.sin(theta2)

        xacc = -(self.g*self.l1**2*self.m1**2*(self.I2 + self.l2**2*self.m2)*np.sin(2*theta1)/2 + self.g*self.l2**2*self.m2**2*(self.I1 + self.l1**2*self.m1)*np.sin(2*theta2)/2 + (self.I1 + self.l1**2*self.m1)*(self.I2 + self.l2**2*self.m2)*(force + self.l1*self.m1*theta1_dot**2*np.sin(theta1) + self.l2*self.m2*theta2_dot**2*np.sin(theta2)))/(self.l1**2*self.m1**2*(self.I2 + self.l2**2*self.m2)*np.cos(theta1)**2 + self.l2**2*self.m2**2*(self.I1 + self.l1**2*self.m1)*np.cos(theta2)**2 - (self.I1 + self.l1**2*self.m1)*(self.I2 + self.l2**2*self.m2)*(self.m0 + self.m1 + self.m2))

        theta1_acc = self.l1*self.m1*(self.g*self.l2**2*self.m2**2*(-np.sin(theta1 - 2*theta2) + np.sin(theta1 + 2*theta2))/4 - self.g*(self.l2**2*self.m2**2*np.cos(theta2)**2 - (self.I2 + self.l2**2*self.m2)*(self.m0 + self.m1 + self.m2))*np.sin(theta1) + (self.I2 + self.l2**2*self.m2)*(force + self.l1*self.m1*theta1_dot**2*np.sin(theta1) + self.l2*self.m2*theta2_dot**2*np.sin(theta2))*np.cos(theta1))/(self.l1**2*self.m1**2*(self.I2 + self.l2**2*self.m2)*np.cos(theta1)**2 + self.l2**2*self.m2**2*(self.I1 + self.l1**2*self.m1)*np.cos(theta2)**2 - (self.I1 + self.l1**2*self.m1)*(self.I2 + self.l2**2*self.m2)*(self.m0 + self.m1 + self.m2))

        theta2_acc = self.l2*self.m2*(self.g*self.l1**2*self.m1**2*(np.sin(2*theta1 - theta2) + np.sin(2*theta1 + theta2))/4 - self.g*(self.l1**2*self.m1**2*np.cos(theta1)**2 - (self.I1 + self.l1**2*self.m1)*(self.m0 + self.m1 + self.m2))*np.sin(theta2) + (self.I1 + self.l1**2*self.m1)*(force + self.l1*self.m1*theta1_dot**2*np.sin(theta1) + self.l2*self.m2*theta2_dot**2*np.sin(theta2))*np.cos(theta2))/(self.l1**2*self.m1**2*(self.I2 + self.l2**2*self.m2)*np.cos(theta1)**2 + self.l2**2*self.m2**2*(self.I1 + self.l1**2*self.m1)*np.cos(theta2)**2 - (self.I1 + self.l1**2*self.m1)*(self.I2 + self.l2**2*self.m2)*(self.m0 + self.m1 + self.m2))

        # calculate accelerations
        numerator_0 = (force + self.pm_len1 * s1 * theta1_dot**2 + self.pm_len2 * s2 * theta2_dot**2) * self.inertia_1 * self.inertia_2
        numerator_1 = self.inertia_2 * self.pm_len1**2 * self.g * c1 * s1
        numerator_2 = self.inertia_1 * self.pm_len2**2 * self.g * c2 * s2
        denominator = self.inertia_1 * self.inertia_2 * self.total_mass - (self.inertia_2 * self.pm_len1**2 * c1 * c1 + self.inertia_1 * self.pm_len2**2 * c2 * c2)

        xacc_ = (numerator_0 + numerator_1 + numerator_2) / (denominator)
        theta1_acc = -(self.pm_len1 *( self.g * s1 + c1 * xacc)) / self.inertia_1
        theta2_acc = -(self.pm_len2 *( self.g * s2 + c2 * xacc)) / self.inertia_2
        # update cart position and velocity
        x  = x + self.dt * x_dot + 0.5 * xacc * self.dt**2
        x_dot = x_dot + self.dt * xacc

        # update pole 1 position and angular velocity
        theta1 = theta1 + self.dt * theta1_dot + 0.5 * theta1_acc * self.dt**2
        theta1_dot = theta1_dot + self.dt * theta1_acc

        # update pole 2 position and angular velocity
        theta2 = theta2 + self.dt * theta2_dot + 0.5 * theta2_acc * self.dt**2
        theta2_dot = theta2_dot + self.dt * theta2_acc

        self.state = (x, x_dot, theta1, theta1_dot, theta2, theta2_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                # or theta1 < -self.theta_threshold_radians \
                # or theta1 > self.theta_threshold_radians \
                # or theta2 < -self.theta_threshold_radians \
                # or theta2 > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _linearize(self):
        A = None
        B = None
        return A, B

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
        # self.state[2] += np.pi
        # self.state[4] += np.pi
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen_1 = scale * 1.0 * 100
        polelen_2 = scale * 0.5 * 100
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # cart
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            # pole 1
            l,r,t,b = -polewidth/2,polewidth/2,polelen_1-polewidth/2,-polewidth/2
            pole1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole1.set_color(1.,.0,.0)
            self.pole1_trans = rendering.Transform(translation=(0, axleoffset))
            pole1.add_attr(self.pole1_trans)
            pole1.add_attr(self.carttrans)
            self.viewer.add_geom(pole1)
            # pole 2
            l,r,t,b = -polewidth/2,polewidth/2,polelen_2-polewidth/2,-polewidth/2
            pole2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole2.set_color(.0, 1., .0)
            self.pole2_trans = rendering.Transform(translation=(0, axleoffset))
            pole2.add_attr(self.pole2_trans)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)
            # axle
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.pole1_trans)
            self.axle.add_attr(self.pole2_trans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.pole1_trans.set_rotation(np.pi+x[2])
        self.pole2_trans.set_rotation(np.pi+x[4])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
