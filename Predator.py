import random
import math
from enum import Enum
import numpy as np

scale = 5.0
Default_Engery = 4000

class Speed(): 
    up = 0
    down = -1
    maintain = 1

class Direction(): 
    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v/norm

def limit(v, lim):
  norm = np.linalg.norm(v)
  if norm == 0:
    return v
  return np.multiply(normalize(v), lim)

class Predator:
  radius = 9
  def __init__(self, x, y, generation):

    self.timeframe = 0
    self.age = 0 
    self.generation = generation

    #position
    self.loc = np.array([float(x), float(y)])
    self.prevLoc = np.array([float(x), float(y)])

    # velocity
    self.vel = 1.0
    self.acc = 0.1
    self.direction = Direction.N
    self.direction_text = "N"
    self.speed = Speed.up
    self.speed_text = "speed up"
    self.maxSpeed = 50/scale # 49.7 mph 

    self.mass = 440 # 441lb
    self.captureRadius = 10

    #default tagert prey index and closest prey index
    self.target_idx = 0
    self.closest_idx = 0

    #set default engery and target prey
    self.energy = Default_Engery 
    self.targetPrey = None

    self.touching = None
    self.state = 0

    # thresholds for deciding an action
    self.move_threshold = 0

  def getNNWInputList(self):
    nnlist = []
    nnlist.append(np.linalg.norm(self.vel))
    nnlist.append(self.energy/Default_Engery)
    nnlist.append(self.state)
    return nnlist

  def update(self, predators, preys, info):
    self.direction = info[1]
    self.speed = info[0]
    self.state = self.speed

    # Update Direction
    orientation = np.array([0., 0.])
    if self.direction == 0:
      orientation = np.array([0., -1.0])
      self.direction_text = "N"
    elif self.direction == 1:
      orientation = np.array([1.0, -1.0])
      self.direction_text = "NE"
    elif self.direction == 2:
      orientation = np.array([1.0, 0.])
      self.direction_text = "E"
    elif self.direction == 3:
      orientation = np.array([1.0, 1.0])
      self.direction_text = "SE"
    elif self.direction == 4:
      orientation = np.array([0., 1.0])
      self.direction_text = "S"
    elif self.direction == 5:
      orientation = np.array([-1.0, 1.0])
      self.direction_text = "SW"
    elif self.direction == 6:
      orientation = np.array([-1.0, 0.])
      self.direction_text = "W"
    elif self.direction == 7:
      orientation = np.array([-1.0, -1.0])
      self.direction_text = "NW"
    orientation = normalize(orientation)

    # Update Acc
    if self.speed == 0:
      self.acc = 10/scale
      self.speed_text = "speed up"
    elif self.speed == -1:
      self.acc = -20/scale
      self.speed_text = "slow down"
    elif self.speed == 1:
      self.acc = 0
      self.speed_text = "maintain"

    # Update Speed
    prevVel = self.vel
    self.vel += self.acc
    if self.vel >= self.maxSpeed:
      self.vel = self.maxSpeed
    elif self.vel <= 0:
      self.vel = 0.0

    # Update Energy
    self.consumeEnergy(self.vel)

    # Update Location
    self.prevLoc = self.loc
    self.loc += orientation*self.vel

    self.age += 1
    self.timeframe += 1
    return self.capturePrey(preys)

  def consumeEnergy(self, v):
    self.energy -= (v*v)
    if self.energy < 0:
      self.energy = 0

  # returns a child with a genetic combination of neural net weights of 2 parents ----  Not in Used
  def mate(self, other):
    child = Predator(0,0)
    child.generation = min(self.generation, other.generation) + 1
    # inherit parents connection weights
    for i in range(0,len(self.net.params)):
      if random.random() > .05:
         child.net.params[i] = random.choice([self.net.params[i], other.net.params[i]])
    return child
 

  def capturePrey(self, preys):
    x = self.prevLoc[0]
    y = self.prevLoc[1]
    futureX = self.loc[0]
    futureY = self.loc[1]

    if x >= futureX:
      x += self.captureRadius
      futureX -= self.captureRadius
    else:
      x -= self.captureRadius
      futureX += self.captureRadius  

    if y >= futureY:
      y += self.captureRadius
      futureY -= self.captureRadius
    else:
      y -= self.captureRadius
      futureY += self.captureRadius    

    for prey in preys:
      preyX = prey.loc[0]
      preyY = prey.loc[1] 
      if (x <= preyX and futureX >= preyX) or (x >= preyX and futureX <= preyX):
        if (y <= preyY and futureY >= preyY) or (y >= preyY and futureY <= preyY):
          return prey
    return None
