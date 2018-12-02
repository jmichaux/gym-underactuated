"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Modified from OpenAI to match the cartpole system implemented by
Russ Tedrake in his awesome book Underactuated Robotics.  Unlike
Russ' book, here we are using moment of inertia
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from os import path

class InvertedPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, masscart=1.0, masspole=0.1, total_length=1.0, tau=0.02, task="balance"):
        # set task
        self.task = task
        self.g = self.gravity = 9.8
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = (self.masspole + self.masscart)
        self.L = total_length
        self.l = self.length =self.L / 2
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = tau
        self.I = (1/12) * self.masspole * (2*self.length)**2
        self.I = 0
        self.inertial = self.I + self.masspole * self.length**2
        self.b = 0.0
        self.n_coords = 2

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

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
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # get state
        x, theta, x_dot, theta_dot = self.state
        theta = self._unwrap_angle(theta)

        # clip torque, update dynamics
        u = np.clip(action, -self.force_mag, self.force_mag)
        xacc, thetaacc = self._dyn(u)

        # integrate
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot + 0.5 * self.tau**2 * thetaacc
        self._unwrap_angle(theta)
        theta_dot = theta_dot + self.tau * thetaacc

        # update state
        self.state = np.array([x, theta, x_dot,theta_dot])

        # check if done
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < np.pi - self.theta_threshold_radians \
                or theta > np.pi + self.theta_threshold_radians
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

        return self.state, reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if self.task == "balance":
            self.state[1] += np.pi
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(.62, .62, .62)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.88, .4, .4) 
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.26, .26, .26)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        # theta with respect to positive vertical (clockwise for positive theta)
        self.poletrans.set_rotation(x[1] + np.pi)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


    def _dyn(self, force):
        """
        Calculate the accelerations
        """
        x, theta, x_dot, theta_dot = self.state
        theta = self._unwrap_angle(theta)
        f = force
        b = self.b
        I = self.inertial
        d0 = self.total_mass
        d1 = self.polemass_length * np.cos(theta)
        d2 = self.polemass_length * np.sin(theta) * theta_dot**2
        d3 = self.polemass_length * np.sin(theta) * self.g

        xacc = ((f + d2) *  I + d1 * d3) / (d0 * I - d1**2)
        thetaacc = -(d3 + d1 * xacc) / I
        return xacc, thetaacc

    def _dyn2(self, state, torque):
        """
        Calculate the accelerations
        """
        u = torque
        pos = state[:self.n_coords]
        vel = state[self.n_coords:]

        Minv = self._Minv(pos)
        B = self._B()
        C = self._C(pos, vel)
        G = self._G(pos)
        acc = np.dot(Minv, B.dot(u) - C.dot(vel.reshape((2,1))) - G)
        return acc.flatten()

    def _M(self, pos):
        """
        Inertial Mass matrix
        
        Arguments
            pos (list):  
                x - position of cart
                th - angle of pole
        """
        x, th = pos

        th = self._unwrap_angle(th)
        I = self.inertial
        d0 = self.total_mass
        d1 = self.polemass_length * np.cos(th)

        mass_matrix = np.array([[d0, d1],
                               [d1, I]])
        return mass_matrix

    def _Minv(self, pos):
        """
        Invert the mass matrix
        """
        return np.linalg.inv(self._M(pos))

    def _C(self, pos, vel):
        """
        Coriolis matrix
        """
        x, theta = pos
        x_dot, theta_dot = vel
        theta = self._unwrap_angle(theta)
        d1 = self.polemass_length * theta_dot * np.sin(theta)
        return np.array([
                        [0, -d1],
                        [0, 0]
                        ])

    def _G(self, pos):
        """
        Gravitional matrix
        """
        x, theta = pos
        d3 = self.polemass_length * np.sin(theta) * self.g

        return np.array([[0], 
                        [d3]])

    def _G2(self, pos):
        """
        Gravitional matrix
        """
        return np.array([[0], 
                        [self.polemass_length * np.sin(pos[1]) * self.g]])

    def _B(self):
        """
        Force matrix
        """
        return np.array([[1], [0]])

    def _linearize(self, state):
        """
        Linearize the system dynamics around a given point
        """
        pos = state[0:2]
        return self._Alin(state), self._Blin(pos)

    def _Alin(self, state):
        pos = state[:self.n_coords]
        vel = state[self.n_coords:]
        ul = np.zeros((self.n_coords, self.n_coords))
        ur = np.eye(self.n_coords)
        Minv = self._Minv(pos)
        deltaG = self._deltaG()
        C = self._C(pos, vel)
        ll = -np.dot(Minv, deltaG)
        lr = -np.dot(Minv, C)
        return np.block([[ul, ur],
                        [ll, lr]])

    def _Blin(self, pos):
        Z = np.dot(self._Minv(pos), self._B())
        return np.block([
                        [np.zeros_like(Z)],
                        [Z]
                        ])

    def _deltaG(self):
        m = self.masspole
        g = self.g
        l = self.l
        return np.array([
                        [0, 0],
                        [0, -m*g*l]
                        ])
        
    # def _jacG(self, state):

    #     self.gravity_jacobian()
    #     return 



    # def _A(self, state):
    #     Minv = self._Minv(state)
    #     ul = np.zeros((self.n_coords, self.n_coords))
    #     ur = np.eye(self.n_coords)
    #     ll = - np.dot(Minv, self._jacG(state))
    #     lr = -np.dot(Minv, self._C(state))
    #     return np.block([[ul, ur],
    #                     [ll, lr]])

    # def _B(self, state):
    #     Z = np.dot(self._Minv(state), self._F)
    #     return np.block([
    #                     [np.zeros_like(Z)],
    #                     [Z]
    #                     ])

    
    # def total_energy(self):
    #     x, th, xdot thdot = self.state

    #     m = self.m
    #     g = self.g
    #     l = self.l
    #     c = np.cos(th)
    #     return 0.5 * m * l**2 * thdot**2 - m*g*l*c

    def kinetic_energ(self, pos, vel):
        return

    def potential_energy(self, pos):
        return

    def desired_energy(self):
        return self.potential_energy()

    def desired_energy(self):
        m = self.m
        g = self.g
        l = self.l
        return m*g*l

    def _unwrap_angle(self, theta):
        sign = (theta >=0)*1 - (theta < 0)*1
        theta = np.abs(theta) % (2 * np.pi)
        return sign*theta

    def integrate(self):
        """
        Integrate the equations of motion
        """
        raise NotImplementedError()


