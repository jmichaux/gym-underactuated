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

    def __init__(self, task="balance", initial_state=None):
        # set task
        self.task = task

        self.initial_state = initial_state
        
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
        self.n_coords = 3

        # precompute the jacobian of the dynamics
        self.jacobian = self._jacobian()

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

    def reset(self):
        if self.initial_state:
            self.state = self.initial_state
        else:
            self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(6,))
            if self.task == "balance":
                self.state[1] += np.pi
                self.state[2] += np.pi
        self.steps_beyond_done = None
        return np.array(self.state)

    def is_done(self):
        x, th1, th2 = self.state[:3]
        if self.task == "balance":
            done =  x < -self.x_threshold \
                    or x > self.x_threshold \
                    or th1 < np.pi - self.theta_threshold_radians \
                    or th1 > np.pi + self.theta_threshold_radians \
                    or th2 < np.pi - self.theta_threshold_radians \
                    or th2 > np.pi + self.theta_threshold_radians
        else:
            pass
        return bool(done)

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # get state
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = self.state
        x, th1, th2, x_dot, th1_dot, th2_dot = self.state

        # clip torque, update dynamics
        # force = self.force_mag if action==1 else -self.force_mag
        u = np.clip(action, -self.force_mag, self.force_mag)
        acc = self._accels(anp.array([x, th1, th2, x_dot, th1_dot, th2_dot, u]))

        # integrate
        xacc, th1_acc, th2_acc = acc

        # update cart position and velocity
        x  = x + self.dt * x_dot + 0.5 * xacc * self.dt**2
        x_dot = x_dot + self.dt * xacc

        # update pole 1 position and angular velocity
        th1 = th1 + self.dt * th1_dot + 0.5 * th1_acc * self.dt**2
        th1_dot = th1_dot + self.dt * th1_acc

        # update pole 2 position and angular velocity
        th2 = th2 + self.dt * th2_dot + 0.5 * th2_acc * self.dt**2
        th2_dot = th2_dot + self.dt * th2_acc

        # update state
        th1 = self._unwrap_angle(th1)
        th2 = self._unwrap_angle(th2)
        self.state = np.array([x, th1, th2, x_dot, th1_dot, th2_dot])
        
        # c1 = math.cos(theta1)
        # s1 = math.sin(theta1)
        # c2 = math.cos(theta2)
        # s2 = math.sin(theta2)

        # xacc = -(self.g*self.l1**2*self.m1**2*(self.I2 + self.l2**2*self.m2)*np.sin(2*theta1)/2 + self.g*self.l2**2*self.m2**2*(self.I1 + self.l1**2*self.m1)*np.sin(2*theta2)/2 + (self.I1 + self.l1**2*self.m1)*(self.I2 + self.l2**2*self.m2)*(force + self.l1*self.m1*theta1_dot**2*np.sin(theta1) + self.l2*self.m2*theta2_dot**2*np.sin(theta2)))/(self.l1**2*self.m1**2*(self.I2 + self.l2**2*self.m2)*np.cos(theta1)**2 + self.l2**2*self.m2**2*(self.I1 + self.l1**2*self.m1)*np.cos(theta2)**2 - (self.I1 + self.l1**2*self.m1)*(self.I2 + self.l2**2*self.m2)*(self.m0 + self.m1 + self.m2))

        # theta1_acc = self.l1*self.m1*(self.g*self.l2**2*self.m2**2*(-np.sin(theta1 - 2*theta2) + np.sin(theta1 + 2*theta2))/4 - self.g*(self.l2**2*self.m2**2*np.cos(theta2)**2 - (self.I2 + self.l2**2*self.m2)*(self.m0 + self.m1 + self.m2))*np.sin(theta1) + (self.I2 + self.l2**2*self.m2)*(force + self.l1*self.m1*theta1_dot**2*np.sin(theta1) + self.l2*self.m2*theta2_dot**2*np.sin(theta2))*np.cos(theta1))/(self.l1**2*self.m1**2*(self.I2 + self.l2**2*self.m2)*np.cos(theta1)**2 + self.l2**2*self.m2**2*(self.I1 + self.l1**2*self.m1)*np.cos(theta2)**2 - (self.I1 + self.l1**2*self.m1)*(self.I2 + self.l2**2*self.m2)*(self.m0 + self.m1 + self.m2))

        # theta2_acc = self.l2*self.m2*(self.g*self.l1**2*self.m1**2*(np.sin(2*theta1 - theta2) + np.sin(2*theta1 + theta2))/4 - self.g*(self.l1**2*self.m1**2*np.cos(theta1)**2 - (self.I1 + self.l1**2*self.m1)*(self.m0 + self.m1 + self.m2))*np.sin(theta2) + (self.I1 + self.l1**2*self.m1)*(force + self.l1*self.m1*theta1_dot**2*np.sin(theta1) + self.l2*self.m2*theta2_dot**2*np.sin(theta2))*np.cos(theta2))/(self.l1**2*self.m1**2*(self.I2 + self.l2**2*self.m2)*np.cos(theta1)**2 + self.l2**2*self.m2**2*(self.I1 + self.l1**2*self.m1)*np.cos(theta2)**2 - (self.I1 + self.l1**2*self.m1)*(self.I2 + self.l2**2*self.m2)*(self.m0 + self.m1 + self.m2))

        # # calculate accelerations
        # numerator_0 = (force + self.pm_len1 * s1 * theta1_dot**2 + self.pm_len2 * s2 * theta2_dot**2) * self.inertia_1 * self.inertia_2
        # numerator_1 = self.inertia_2 * self.pm_len1**2 * self.g * c1 * s1
        # numerator_2 = self.inertia_1 * self.pm_len2**2 * self.g * c2 * s2
        # denominator = self.inertia_1 * self.inertia_2 * self.total_mass - (self.inertia_2 * self.pm_len1**2 * c1 * c1 + self.inertia_1 * self.pm_len2**2 * c2 * c2)

        # xacc_ = (numerator_0 + numerator_1 + numerator_2) / (denominator)
        # theta1_acc = -(self.pm_len1 *( self.g * s1 + c1 * xacc)) / self.inertia_1
        # theta2_acc = -(self.pm_len2 *( self.g * s2 + c2 * xacc)) / self.inertia_2
        # # update cart position and velocity
        # x  = x + self.dt * x_dot + 0.5 * xacc * self.dt**2
        # x_dot = x_dot + self.dt * xacc

        # # update pole 1 position and angular velocity
        # theta1 = theta1 + self.dt * theta1_dot + 0.5 * theta1_acc * self.dt**2
        # theta1_dot = theta1_dot + self.dt * theta1_acc

        # # update pole 2 position and angular velocity
        # theta2 = theta2 + self.dt * theta2_dot + 0.5 * theta2_acc * self.dt**2
        # theta2_dot = theta2_dot + self.dt * theta2_acc

        # self.state = (x, x_dot, theta1, theta1_dot, theta2, theta2_dot)
        # done =  x < -self.x_threshold \
        #         or x > self.x_threshold \
        #         # or theta1 < -self.theta_threshold_radians \
        #         # or theta1 > self.theta_threshold_radians \
        #         # or theta2 < -self.theta_threshold_radians \
        #         # or theta2 > self.theta_threshold_radians
        done = self.is_done()

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

    def _accels(self, vec):
        """
        Calculate the accelerations
        """
        force = vec[-1]
        pos = vec[:self.n_coords]
        vel = vec[self.n_coords:-1]
        state = vec[:-1]
        Minv = self._Minv(pos)
        B = self._B()
        C = self._C(state)
        G = self._G(pos)
        acc = anp.dot(Minv, anp.dot(B, force) - anp.dot(C, vel.reshape((self.n_coords, 1))) - G)
        return acc.flatten()

    def _F(self, vec):
        """
        Return derivative of state-space vector
        """
        qd = vec[self.n_coords:-1]
        qdd = self._accels(vec)
        return anp.array(list(qd) + list(qdd))

    def _M(self, pos):
        """
        Inertial Mass matrix
        """
        x, th1, th2 = pos
        d0 = self.total_mass
        d1 = self.pm_len1 * anp.cos(th1)
        d2 = self.pm_len2 * anp.cos(th2)
        I1 = self.inertia_1
        I2 = self.inertia_2

        mass_matrix = anp.array([[d0, d1, d2],
                               [d1, I1, 0],
                               [d2, 0, I2]])
        return mass_matrix

    def _C(self, state):
        """
        Coriolis matrix
        """
        x, th1, th2 xdot, th1_dot, th2_dot = state
        d1 = self.pm_len1 * th1_dot * anp.sin(th1)
        d2 = self.pm_len2 * th2_dot * anp.sin(th2)
        return anp.array([[0, -d1, -d2],
                        [0, 0, 0],
                        [0, 0, 0],
                        ])

    def _G(self, pos):
        """
        Gravitional matrix
        """
        x, th1, th2 = pos
        g1 = self.pm_len1 * anp.sin(th1) * self.g
        g2 = self.pm_len2 * anp.sin(th2) * self.g
        return anp.array([[0], 
                        [g1],
                        [g2]])

    def _B(self):
        """
        Force matrix
        """
        return anp.array([[1], [0], [0]])

    def _jacobian(self):
        """
        Return the Jacobian of the full state equation
        """
        return jacobian(self._F)

    def _linearize(self, vec):
        """
        Linearize the dynamics by first order Taylor expansion
        """
        f0 = self._F(vec)
        arr = self.jacobian(vec)
        A = arr[:, :-1]
        B = arr[:, -1].reshape((2 * self.n_coords, 1))
        return f0, A, B

    def _Minv(self, pos):
        """
        Invert the mass matrix
        """
        return anp.linalg.inv(self._M(pos))

    def total_energy(self, state):
        pos = state[:self.n_coords]
        vel = state[self.n_coords:]
        return self.kinetic_energy(pos, vel) + self.potential_energy(pos)

    def kinetic_energy(self, pos, vel):
        return

    def potential_energy(self, pos):
        return

    def _unwrap_angle(self, theta):
        sign = (theta >=0)*1 - (theta < 0)*1
        theta = anp.abs(theta) % (2 * anp.pi)
        return sign*theta

    def integrate(self):
        """
        Integrate the equations of motion
        """
        raise NotImplementedError()

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
