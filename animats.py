#!/usr/bin/python
import pickle
import random
import math
from enum import Enum
import numpy as np
import sys

from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
Default_Engery = 100

class Behavior(Enum): 
    stay = 0
    stalk  = -1
    hunt = 1


class Environment:
  def __init__(self, num_predator, width, height, filename):
    # training mode (foods everywhere)
    self.training_mode = False
    # environment
    self.width = width
    self.height = height
    # print self.width/2, self.height/2
    self.num_predator = num_predator
    # record log
    self.log = []
    self.moveLog = []
    # save state
    self.filename = filename

    # animats
    self.deaths = []
    self.predators = []
    self.capturedPrey = []
    self.placeRadius = 200;
    saved_states = self.load()


    # prey
    self.preys = []
    num_preys = 2
    for i in range(num_preys):
      p = Prey(random.random() * 360, random.random() * 360)
      self.preys.append(p)

    for i in range(self.num_predator):
      # print i
      pos = self.findSpace(i, self.placeRadius, Predator.radius)
      if len(saved_states) > 0:
        a = saved_states.pop(0)
        a.x = pos[0]
        a.y = pos[1]
      else:
        a = Predator(pos[0], pos[1])
        a.generation = 1
      self.predators.append(a)
  # prey

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
      prey.update(self.preys)

    # update each animat
    for animat in self.predators:
      # Sight
      animat.sees = self.line_of_sight(animat)
      # Touch
      step = 3
      step_x = int(math.cos(animat.direction*math.pi / 180) * step)
      step_y = int(math.sin(animat.direction*math.pi / 180) * step)
      animat.touching = self.collision(animat.loc[0] + step_x, animat.loc[1] + step_y, Predator.radius, animat)
      # Capture
      capture = self.capture(animat.loc[0] + step_x, animat.loc[1] + step_y, Predator.radius, animat)
      # update
      animat.update(self.predators, self.preys)

      # moviing
      animat.loc[0] = step_x + animat.loc[0]
      animat.loc[1] = step_y + animat.loc[1]

      # DEATH 
      if animat not in self.deaths \
      and (animat.energy < 0):
	       self.deaths.append(animat)
        

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

  def capture(self, x, y, radius, without=None):
    # check if captured
    if (y + radius) > self.height or (x + radius) > self.width  \
    or (x - radius) < 0 or (y - radius) < 0:
      return self
    # check animat-animat collision
    predators = list(self.predators)
    preys = list(self.preys)
    prey_idx = []
    if without:
      predators.remove(without)
    for animat in predators:
      count = 0
      for prey in preys:
        if (x - animat.loc[0])**2 + (y - animat.loc[1])**2 <= Predator.radius**2:
          prey_idx.append(count)
        count += 1
    captured = []
    for i in range(0,len(prey_idx)):
      captured.append(self.preys[prey_idx[i]])
    for i in range(0,len(prey_idx)):
      self.preys.remove(captured[i])
      print "remove: " + str(i) + " prey" 
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
    self.mag = 3

    self.loc = [float(x), float(y)]
    self.vel = [10., 0.]
    self.acc = [0., 0.]
    self.maxForce = 10
    self.mass = 12
    self.r = self.mass / 2    
    #for testing
    self.width = 1000
    self.height = 700
    #-----------end of testing
  def update(self, preys):
    self.preyForce(preys)
    self.vel = np.add(self.vel, self.acc)
    self.loc = np.add(self.loc, self.vel)
    self.acc = [0., 0.]
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
    force /= self.mass
    self.acc = np.add(self.acc, force)

  def avoidForce(self, preys):
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

  def approachForce(self, preys):
    count = 0
    locSum = [0., 0.]
    for p in preys:
      approachRadius = self.mass + (60*self.mag)
      dist = np.subtract(p.loc, self.loc)
      d = np.linalg.norm(dist)
      if d != 0 and d < approachRadius:
        locSum = np.add(locSum, p.loc)
        count += 1
    if count > 0:
      locSum /= count
      approachVec = np.subtract(locSum, self.loc)
      #np.linalg.norm()
      self.applyF(approachVec)

  def alignForce(self, preys):
    count = 0
    velSum = [0., 0.]
    for p in preys:
      alignRadius = self.mass + (100*self.mag)
      dist = np.subtract(p.loc, self.loc)
      d = np.linalg.norm(dist)
      if d != 0 and d < alignRadius:
        velSum = np.add(velSum, p.vel)
        count += 1
      if count > 0:
        velSum /= count
        alignVec = velSum
        self.applyF(alignVec)

  def preyForce(self, preys):
    self.avoidForce(preys)
    self.approachForce(preys)
    self.alignForce(preys)

'''
    if self.loc[0] <= 0:
      self.loc[0] = self.width
    elif self.loc[0] > self.width:
      self.loc[0] = 0

    if self.loc[1] <= 0:
      self.loc[1] = self.height
    elif self.loc[1] > self.height:
      self.loc[1] = 0     
'''

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
    self.mag = 3
    #position
    self.loc = [float(x), float(y)]
    # velocity
    self.vel = [10., 0.]

    self.acc = [0., 0.]
    self.maxForce = 10
    self.mass = 12
    self.r = self.mass / 2   

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
    self.vel = np.add(self.vel, self.acc)
    self.loc = np.add(self.loc, self.vel)
    self.acc = [0., 0.]
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
    ''' self is lion 0
     lion 1 position - self position
     lion 2 position - self position
     zibra 1 position - self position
     zibra 2 position - self postion
     self Behavior
     lion 1 Behavior
     lion 2 Behavior
     zibra 1 behavior
     zibra 2 behavior
    '''

    sensors = ()
    '''decision = self.net.activate(sensors)'''
    # get a little hungry no matter what
    self.age += 1
    self.get_hungry(self.behavior)
    self.timeframe += 1
    # move forward
    #self.wants_to_move = (decision[0] > self.move_threshold)
    # rotate left 
    #self.direction -= decision[1]
    # rotate right 
    #self.direction += decision[2]

  def get_hungry(self, action):
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
    force /= self.mass
    self.acc = np.add(self.acc, force)

  def approachForce(self, preys):
    count = 0
    locSum = [0., 0.]
    approachRadius = self.mass + 260
    captureRadius = 3.0
    min_dist = sys.float_info.max
    min_idx = -1
    #find the closest prey
    for p in preys:
      dist = np.subtract(p.loc, self.loc)
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
      approachVec = np.subtract(preys[min_idx].loc,  self.loc)
      self.applyF(approachVec)

  def avoidForce(self, prads):
    count = 0
    locSum = [0., 0.]
    for p in prads:
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

  def alignForce(self, prads):
    count = 0
    velSum = [0., 0.]
    for p in prads:
      alignRadius = self.mass + (100*self.mag)
      dist = np.subtract(p.loc, self.loc)
      d = np.linalg.norm(dist)
      if d != 0 and d < alignRadius:
        velSum = np.add(velSum, p.vel)
        count += 1
      if count > 0:
        velSum /= count
        alignVec = velSum
        self.applyF(alignVec)

  def PredForce(self, preys, prads):
    self.approachForce(preys)
    self.avoidForce(prads)
    self.alignForce(prads)
  def record(self):
    #record current locations
    return