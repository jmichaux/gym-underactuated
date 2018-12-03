# Underactuated Systems (under development)
This repository contains modified OpenAI gym environments for continuous control.  The purpose of this project is to create environments that can be used to prototype/experiment with optimal control and reinforcement learning algorithms.   

# Enviroments
So far, working environments include the Pendulum, Inverted Pendulum (Cartpole), and Dual Inverted Pendulum.  Environments currently under development include the Double Inverted Pendulum, Acrowheel (wheeled inverted pendulum + acrobot), a Raibert-style hopper, and two- and three-link bipeds.

### Dynamics
All of the equations of motion have been derived by hand.  Using the standard manipulator form for robot dynamics, we keep track of the inertial mass matrix **M**, the Coriolis matrix **C**, gravity vector **G**, and a vector/array **B** that maps torque inputs to generalized coordinates.  By using the manipulator equations, we can easily obtain the linearized dynamics equations.

### Numerical Integration
We currently use a simple forward-Euler integration scheme.

# Contributions
Contributions are greatly appreciated.
