import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from pendulum import PendulumEnv
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
Q = np.array([[1,0], [0,10]])
R = np.array([[0.001]])

# Compute gain matrix K
K = lqr(A,B,Q,R)

# Run environment
i = 0
while not done:
    env.render()
    if i >= 1:
        action = -np.matmul(K, state.reshape(2) - np.array([np.pi, 0]))
    else:
        action = -np.matmul(K, state - np.array([np.pi, 0]))
    state, _, done, _ = env.step(action)
    i += 1
env.close()
