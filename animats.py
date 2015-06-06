#!/usr/bin/python
import pickle
import random
import math
from enum import Enum
import numpy as np
import sys

from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

preyFleeing = 0
Default_Engery = 100

class Behavior(Enum): 
    stay = 0
    stalk  = -1
    hunt = 1

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


class Environment:
  def __init__(self, num_predator, num_prey, width, height, filename):

    # environment
    self.width = width
    self.height = height
    # print self.width/2, self.height/2
    self.num_predator = num_predator
    self.num_prey = num_prey
    # record log
    self.log = []
    self.moveLog = []
    # save state
    self.filename = filename

    # predators
    self.deaths = []
    self.predators = []
    # self.placeRadius = 200;
    saved_states = self.load()

    # prey
    self.preys = []

    # create prey instances in middle
    for i in range(self.num_prey):
      pos = self.findSpace(i, 50, Predator.radius)
      p = Prey(pos[0], pos[1])
      self.preys.append(p)

    # create predator instances
    for i in range(self.num_predator):
      # print i
      pos = self.findSpace(i, 200, Predator.radius)
      if len(saved_states) > 0:
        a = saved_states.pop(0)
        a.x = pos[0]
        a.y = pos[1]
      else:
        a = Predator(pos[0], pos[1])
        a.generation = 1
      self.predators.append(a)

  # line of sight
  def line_of_sight(self, animat):
    step_x = int(math.cos(animat.direction*math.pi / 180) * 10)
    step_y = int(math.sin(animat.direction*math.pi / 180) * 10)
    new_x = animat.loc[0] + step_x
    new_y = animat.loc[1] + step_y
    sees = None
    while not sees:
      new_x += step_x
      new_y += step_y
      sees = self.collision(new_x, new_y, Predator.radius, animat)
    return sees

  def findSpace(self, count, placeRadius, AnimateRadius):
    noCoverDegree = 20
    degree = random.randrange(noCoverDegree , 360.0/self.num_predator - noCoverDegree)  # random degree
    degree = degree + count * 360.0/self.num_predator
    print degree
    degree = math.radians(degree) #Convert angle from degrees to radians.
    radius = random.randrange(placeRadius, placeRadius + 20)
    x = math.cos(degree) * radius
    y = math.sin(degree) * radius

    centerX = self.width / 2
    centerY = self.height /2
    # print math.cos(math.radians(360)), math.sin(math.radians(270))
    # print centerX, centerY, x, y, degree
    x = centerX + x
    y = centerY + y
    return (x, y)


  def update(self):
    # if an animat died, the two fittest predators mate
    while len(self.deaths) > 0:
      self.predators.remove(self.deaths.pop(0))
      print "die"
    
    # update each prey
    for prey in self.preys:
      prey.update(self.preys, self.predators)

    # update each predator
    for pred in self.predators:
      # Sight
      #pred.sees = self.line_of_sight(pred)
      # Touch
      #step = 3
      #step_x = int(math.cos(pred.direction*math.pi / 180) * step)
      #step_y = int(math.sin(pred.direction*math.pi / 180) * step)
      #pred.touching = self.collision(pred.loc[0] + step_x, pred.loc[1] + step_y, Predator.radius, pred)
      # update
      pred.update(self.predators, self.preys)

      # moviing
      #pred.loc[0] = step_x + pred.loc[0]
      #pred.loc[1] = step_y + pred.loc[1]

      # DEATH 
      if pred not in self.deaths and (pred.energy < 0):
        self.deaths.append(pred)
        

  def collision(self, x, y, radius, without=None):
    # check wall collision
    if (y + radius) > self.height or (x + radius) > self.width  \
    or (x - radius) < 0 or (y - radius) < 0:
      return self

    # check animat-animat collision
    predators = list(self.predators)
    if without:
      predators.remove(without)
    for animat in predators:
      if (x - animat.loc[0])**2 + (y - animat.loc[1])**2 <= Predator.radius**2:
        return animat
    # no collision
    return None

  # load animat states
  def load(self):
    if self.filename == "":
      return []
    try:
      f = open(self.filename, 'r')
      predators = pickle.load(f)
      f.close()
      return predators
    except:
      print "Could not load file " + self.filename
      return []

  # save neural net states
  def save(self):
    if self.filename != "":
      f = open(self.filename, 'w')
      pickle.dump(self.predators, f)
      f.close()

# prey
class Prey:
  def __init__(self, x, y):

    self.loc = np.array([float(x), float(y)])
    self.vel = np.array([0., 0.])
    self.acc = np.array([0., 0.])
    self.maxForce = 3
    self.mass = 10 
    self.repelRadius = 50  
  
  def update(self, preys, preds):
    self.repelForce(preds, self.repelRadius)
    if preyFleeing == 1:
      self.preyFleeForce(preys)
      self.vel += self.acc
      self.loc += self.vel      
    else: # if prey idle, random move
      self.loc += np.array([random.random()*4 - 2, random.random()*4 - 2])
    self.acc = np.array([0., 0.])
    '''#for testing
    if self.loc[0] <= 0:
      self.loc[0] = self.width
    elif self.loc[0] > self.width:
      self.loc[0] = 0

    if self.loc[1] <= 0:
      self.loc[1] = self.height
    elif self.loc[1] > self.height:
      self.loc[1] = 0 
    #------------end of testing'''
  def applyF(self, force):
    # F = ma (a = F/m)
    a = force / self.mass
    self.acc += a

  # avoid the average position of other boids
  def avoidForce(self, preys):
    count = 0
    locSum = np.array([0., 0.])
    for otherPrey in preys:
      separation = self.mass + 20
      dist = otherPrey.loc - self.loc
      d = np.linalg.norm(dist)
      if d != 0 and d < separation:
        locSum += otherPrey.loc
        count += 1
    if count > 0:
      locSum /= count # average loc
      avoidVec = self.loc - locSum
      avoidVec = limit(avoidVec, self.maxForce*2.5)
      self.applyF(avoidVec)

  # cohesion
  def approachForce(self, preys):
    count = 0
    locSum = np.array([0., 0.])
    for otherPrey in preys:
      approachRadius = self.mass + 60
      dist = otherPrey.loc - self.loc
      d = np.linalg.norm(dist)
      if d != 0 and d < approachRadius:
        locSum += otherPrey.loc
        count += 1
    if count > 0:
      locSum /= count
      approachVec = locSum - self.loc
      approachVec = limit(approachVec, self.maxForce)
      self.applyF(approachVec)

  # align
  def alignForce(self, preys):
    count = 0
    velSum = np.array([0., 0.])
    for otherPrey in preys:
      alignRadius = self.mass + 100
      dist = otherPrey.loc - self.loc
      d = np.linalg.norm(dist)
      if d != 0 and d < alignRadius:
        velSum += otherPrey.vel
        count += 1
      if count > 0:
        velSum /= count
        alignVec = velSum
        alignVec = limit(alignVec, self.maxForce)
        self.applyF(alignVec)

  def repelForce(self, preds, r):
    for pred in preds:
      futurePos = self.loc + self.vel
      dist = pred.loc - futurePos
      d = np.linalg.norm(dist)

      if d <= r:
        # change prey state
        global preyFleeing
        if preyFleeing == 0:
          preyFleeing = 1

        repelVec = self.loc - pred.loc
        repelVec = normalize(repelVec)
        repelVec *= (self.maxForce * 5)
        self.applyF(repelVec)

  def preyFleeForce(self, preys):
    self.avoidForce(preys)
    self.approachForce(preys)
    self.alignForce(preys)

# Animats     
class Predator:
  radius = 30
  def __init__(self, x, y):
    #for testing
    #self.width = 1000
    #self.height = 700
    #-----------end of testing
    self.timeframe = 0
    self.age = 0 # how long does it live

    #position
    self.loc = np.array([float(x), float(y)])
    # velocity
    self.vel = np.array([0., 0.])
    self.acc = np.array([0., 0.])

    self.maxForce = 30
    self.mass = 32 

    #set default engery
    self.energy = Default_Engery 
    self.targetPrey = None

    #set orientation range in (0 - 359 degrees)
    self.direction = 0

    self.touching = None
    self.sees = None
    self.behavior = Behavior.stay


    # neural net
    self.net = FeedForwardNetwork()
    self.net.addInputModule(LinearLayer(9, name='in'))
    self.net.addModule(SigmoidLayer(9, name='hidden'))
    self.net.addOutputModule(LinearLayer(3, name='out'))
    self.net.addConnection(FullConnection(self.net['in'], self.net['hidden']))
    self.net.addConnection(FullConnection(self.net['hidden'], self.net['out']))
    self.net.sortModules()

    # thresholds for deciding an action
    self.move_threshold = 0

  def update(self, predators, preys):
    self.PredForce( preys, predators )
    self.vel += self.acc
    self.loc += self.vel
    self.acc = np.array([0., 0.])
    '''#for testing
    if self.loc[0] <= 0:
      self.loc[0] = self.width
    elif self.loc[0] > self.width:
      self.loc[0] = 0
    if self.loc[1] <= 0:
      self.loc[1] = self.height
    elif self.loc[1] > self.height:
      self.loc[1] = 0 
    if self.timeframe%5 == 0:
      self.record()
    '''


    sensors = ()
    '''decision = self.net.activate(sensors)'''
    # consume energy based on differnt current behavior
    self.age += 1
    self.consumeEnergy(self.behavior)
    self.timeframe += 1


  def consumeEnergy(self, action):
    if action == Behavior.stay:
      self.energy -= 1
    elif action == Behavior.stalk:
      self.energy -= 5
    elif action == Behavior.hunt:
      self.energy -= 10

  # returns a child with a genetic combination of neural net weights of 2 parents
  def mate(self, other):
    child = Predator(0,0)
    child.generation = min(self.generation, other.generation) + 1
    # inherit parents connection weights
    for i in range(0,len(self.net.params)):
      if random.random() > .05:
         child.net.params[i] = random.choice([self.net.params[i], other.net.params[i]])
    return child

  def applyF(self, force):
    # F = ma (a = F/m)
    a = force / self.mass
    self.acc += a

  '''def avoidForce(self, predators):
    count = 0
    locSum = [0., 0.]
    for p in preys:
      separation = self.mass + (20*self.mag)
      dist = np.subtract(p.loc, self.loc)
      d = np.linalg.norm(dist)
      if d != 0 and d < separation:
        locSum = np.add(locSum, p.loc)
        count += 1
    if count > 0:
      locSum /= count
      avoidVec = np.subtract(self.loc, locSum)
      #np.linalg.norm()
      self.applyF(avoidVec)
  '''
  def approachForce(self, preys):
    count = 0
    approachRadius = self.mass + 260
    captureRadius = 3.0
    min_dist = sys.float_info.max
    min_idx = -1
    #find the closest prey
    for prey in preys:
      dist = prey.loc - self.loc
      d = np.linalg.norm(dist)
      if d <= captureRadius:
        print d, count
        print "Capture One Prey"
      #if d < min_dist and d < approachRadius:
      if d < min_dist:
        min_idx = count
      count+=1
    #approach the closest prey
    if min_idx != -1: 
      approachVec = preys[min_idx].loc - self.loc
      approachVec = normalize(approachVec)
      approachVec *= self.maxForce
      self.applyF(approachVec)

  def avoidForce(self, preds):
    count = 0
    locSum = np.array([0., 0.])
    for otherPred in preds:
      separation = self.mass + 20
      dist = otherPred.loc - self.loc
      d = np.linalg.norm(dist)
      if d != 0 and d < separation:
        locSum += otherPred.loc
        count += 1
    if count > 0:
      locSum /= count # average loc
      avoidVec = self.loc - locSum
      avoidVec = limit(avoidVec, self.maxForce)
      self.applyF(avoidVec)

  def PredForce(self, preys, prads):
    self.approachForce(preys)
    #self.avoidForce(prads)

  def record(self):
    #record current locations
    return