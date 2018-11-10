import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import numpy as np
import copy
import math
import random
import pybullet_data
from pprint import pprint
import time
from pybullet_envs.bullet.kuka import Kuka

# cid = p.connect(p.UDP,"192.168.86.100")
cid = p.connect(p.SHARED_MEMORY)
if (cid<0):
  p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()

objects = [p.loadURDF("plane.urdf", 0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000)]
# objects = [p.loadURDF("shape_sorter.urdf", 0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,1.000000)]

# load blocks
obj = [p.loadURDF("dual_inverted_pendulum.urdf", (0,0,2), useFixedBase=True)]
# obj = [p.loadURDF("cartpole.urdf", 0,0,2)]




p.setGravity(0,0,-10)

p.setRealTimeSimulation(1)
ref_time = time.time()

running_time = 360 # seconds
while True:
# while (time.time() < ref_time+running_time):
  p.setGravity(0,0,-10)
  p.stepSimulation()
