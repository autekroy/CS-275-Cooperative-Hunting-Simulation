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

class Prey_multi:
  radius = 9
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
      if self.force[0] != 0 and self.force[1] != 0:
        self.force = np.array([0., 0.]) 
        self.preyFleeForce(preys)
        self.force = normalize(self.force)
        self.vel += (self.force * self.maxForce*0.2)
      self.loc += self.vel      
    else: # if prey idle, random move
      self.loc += np.array([random.random()*4 - 2, random.random()*4 - 2])
    self.force = np.array([0., 0.])

  def applyRF(self, force):
    # F = ma (a = F/m)
    self.force += force

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
        self.applyRF(repelVec)

  # avoid the average position of other boids
  def avoidForce(self, preys):
    count = 0
    locSum = np.array([0., 0.])
    for otherPrey in preys:
      separation = 20
      dist = otherPrey.loc - self.loc
      d = np.linalg.norm(dist)
      if d != 0 and d < separation:
        locSum += otherPrey.loc
        count += 1
    if count > 0:
      locSum /= count # average loc
      avoidVec = self.loc - locSum
      avoidVec = limit(avoidVec, self.maxForce*2.5)
      print "avoidVec = " + str(avoidVec)
      self.applyF(avoidVec)

  # cohesion
  def approachForce(self, preys):
    count = 0
    locSum = np.array([0., 0.])
    for otherPrey in preys:
      approachRadius = 60
      dist = otherPrey.loc - self.loc
      d = np.linalg.norm(dist)
      if d != 0 and d < approachRadius:
        locSum += otherPrey.loc
        count += 1
    if count > 0:
      locSum /= count
      approachVec = locSum - self.loc
      approachVec = limit(approachVec, self.maxForce)
      print "approachVec = " + str(approachVec)
      self.applyF(approachVec)

# align preys
  def alignForce(self, preys):
    count = 0
    velSum = np.array([0., 0.])
    for otherPrey in preys:
      alignRadius = 100
      dist = otherPrey.loc - self.loc
      d = np.linalg.norm(dist)
      if d != 0 and d < alignRadius:
        velSum += otherPrey.vel
        count += 1
      if count > 0:
        velSum /= count
        alignVec = velSum
        alignVec = limit(alignVec, self.maxForce)
        print "alignVec = " + str(alignVec)
        self.applyF(alignVec)

# apply all forces 
  def preyFleeForce(self, preys):
    self.avoidForce(preys)
    self.approachForce(preys)
    self.alignForce(preys)
