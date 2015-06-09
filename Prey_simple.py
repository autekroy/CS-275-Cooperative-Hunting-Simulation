import random
import math
from enum import Enum
import numpy as np
# prey
preyFleeing = 0
scale = 5.0

class Speed(Enum): 
    up = 0
    down = -1
    maintain = 1

class Direction(Enum): 
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

class Prey_simple:
  radius = 7
  def __init__(self, x, y):
    self.loc = np.array([float(x), float(y)])
    self.vel = np.array([0., 0.])
    self.force = np.array([0., 0.])
    self.maxForce = 50/scale # 40mph
    self.mass = 723 # 723.1lb
    self.repelRadius = 100
    self.status = 0
  
  def update(self, preys, preds):
    self.repelForce(preds)
    if preyFleeing == 1:
      self.force = normalize(self.force)
      self.vel = (self.force * self.maxForce)
      self.loc += self.vel      
    else: # if prey idle, random move
      self.loc += np.array([random.random()*4 - 2, random.random()*4 - 2])
    self.force = np.array([0., 0.])

  def applyF(self, force):
    # F = ma (a = F/m)
    self.force += force

  def repelForce(self, preds):
    for pred in preds:
      futurePos = self.loc + self.vel
      dist = pred.loc - futurePos
      d = np.linalg.norm(dist)

      if d <= self.repelRadius:
        # change prey state
        global preyFleeing
        if preyFleeing == 0:
          preyFleeing = 1

        repelVec = normalize(self.loc - pred.loc)
        if d != 0:
          repelVec /= (d*d)
        self.applyF(repelVec)

