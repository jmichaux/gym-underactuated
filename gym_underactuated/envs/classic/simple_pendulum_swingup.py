import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from pendulum_env import PendulumEnv
from scipy.linalg import solve_continuous_are as riccati

def lqr(A, B, Q, R, N=None):
    # Solve Algebraic Riccati Equation
    P = riccati(A, B, Q, R, e=N)
    R_inv = np.linalg.inv(R)
    if N:
        K = np.dot(R_inv, np.dot(B.T, P) + N.T)
    else:
        K = np.dot(R_inv, np.dot(B.T, P))
    return K

# create environment
env = PendulumEnv()

# reset environment
state = env.reset()
done = False

# Get linearized dynamics
A,B = env._linearize(np.pi)

# create cost matrices for LQR
Q = np.array([[1,0], [0,1]])
R = np.array([[0.001]])

# Compute gain matrix K
K = lqr(A,B,Q,R)

# feedback gain for swing-up controller
k = 20
# Run environment
i = 0
balanced = False
while not done:
    env.render()
    th, thdot = state
    if np.abs(th - np.pi) < .1:
        balanced = True
    # TODO: Fix this
    elif np.abs(th - np.pi) > 2:
        balanced = False
    if balanced:
        # TODO: Fix this tomfoolery
        if i >= 1:
            action = -np.matmul(K, state.reshape(2) - np.array([np.pi, 0]))
        else:
            action = -np.matmul(K, state - np.array([np.pi, 0]))
    else:
        # calculate energy
        E = env.total_energy() - env.desired_energy()
        action = -k*thdot*E

    state, _, done, _ = env.step(action)
    i += 1
env.close()
