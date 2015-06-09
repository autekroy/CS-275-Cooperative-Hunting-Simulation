#!/usr/bin/python
import pickle
import random
import math
from enum import Enum
import numpy as np
import sys
import NNW

from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

preyFleeing = 0
Default_Engery = 1000
scale = 5.0

class Behavior(Enum): 
    stay = 0
    stalk  = -1
    hunt = 1

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

class Environment:
  def __init__(self, generation, num_predator, num_prey, width, height, filename):

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
    self.file_fp = None
    if filename!='':
      self.file_fp = open(filename,'w')
    self.timeframe = 0
    # predators

    self.cotarget_idx = None
    self.pred_deaths = []

    self.predators = []

    self.capturedPrey = []
    self.placeRadius = 200;
    self.halt = 0

    saved_states = self.load()

    # prey
    self.preys = []
    self.prey_deaths = []
    for i in range(self.num_prey):
      p = Prey(400+random.random() * 200, 250+random.random() * 200)
      self.preys.append(p)

    # create predator instances
    for i in range(self.num_predator):
      # print i
      pos = self.findSpace(i, 200, 20, Predator.radius)
      if len(saved_states) > 0:
        a = saved_states.pop(0)
        a.x = pos[0]
        a.y = pos[1]
      else:
        a = Predator(pos[0], pos[1], generation)
        a.generation = 1
      self.predators.append(a)
    #---------Neural Network----------#
    #-- Method 1 ---------------------#
    #-- Initial Stage ----------------#
    #-- First Network ---: Speed -----#
    self.speed_net = NNW.NNW(28,32,9)
    #-- Second Network --: Direction -#
    self.dir_net = NNW.NNW(28,40,24)
    #---------------------------------#

  def end_iteration(self):
    return self.halt

  def getNNWInput(self):
    input_vals = []
    for pred in self.predators:
      input_vals = input_vals + pred.getNNWInputList()
      for other in self.predators:
        if pred != other:
          loc = other.loc - pred.loc
          input_vals.append(loc[0])
          input_vals.append(loc[1])
      loc = self.preys[0].loc - pred.loc
      input_vals.append(loc[0])
      input_vals.append(loc[1])
    input_vals.append(self.preys[0].status)
    return input_vals

  def findSpace(self, count, placeRadius, noCoverDegree, AnimateRadius):
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

  def extract_info_from_nnlist(self, nnlist_speed, nnlist_dir):
    info = []
    n_speed_idx_list = np.argsort(nnlist_speed)
    n_speed_top = n_speed_idx_list[::-1][0]
    if n_speed_top == 0:
      info.append(Speed.up)
    elif n_speed_top == -1:
      info.append(Speed.down)
    else:
      info.append(Speed.maintain)
    n_dir_idx_list = np.argsort(nnlist_dir)
    n_dir_top = n_dir_idx_list[::-1][0]
    info.append(n_dir_top)
    return info

  def update(self):
    # if an animat died, the two fittest predators mate
    while len(self.pred_deaths ) > 0:
      self.predators.remove(self.pred_deaths.pop(0))
      print "die"

    while len(self.prey_deaths ) > 0:
      self.preys.remove(self.prey_deaths.pop(0))
      print "capture Prey"


    # update each prey
    for prey in self.preys:
      prey.update(self.preys, self.predators)

    # get the result from NNW
    input_vals = self.getNNWInput()
    nn_out_speed = self.speed_net.activate(input_vals)
    nn_out_dir = self.dir_net.activate(input_vals)

    count = 0
    # update each predator
    for pred in self.predators:
      update_info = self.extract_info_from_nnlist(nn_out_speed[count*3:count*3+3],nn_out_dir[count*8:count*8+8])

      # Capture
      #captured = self.capture(pred.loc[0] , pred.loc[1] , Predator.radius, pred)
      # Update
      pred.update(self.predators, self.preys, update_info)

      # CAPTURE
      deadPrey = pred.capturePrey(self.preys)
      if (deadPrey != None) and (deadPrey not in self.prey_deaths):
        self.prey_deaths.append(deadPrey)
      # DEATH 
      if pred not in self.pred_deaths and (pred.energy < 0):
        self.pred_deaths.append(pred)
      count += 1
 
    #update the target of predator
    if len(self.pred_deaths) > 0 or len(self.prey_deaths) > 0:
      self.halt = 1
      
      r_list = list(input_vals) + list(nn_out_speed) + list(nn_out_dir)
      self.record(r_list)
      self.file_fp.close()
      return

    r_list = list(input_vals) + list(nn_out_speed) + list(nn_out_dir)
    self.update_cotarget()
    self.timeframe += 1
    if self.timeframe == 5:
      self.record(r_list)
      self.timeframe = 0  

  def record(self, r_list):
    for obj in r_list:
      self.file_fp.write(str(obj)+',')
    captured = 0
    left_energy = 0
    distance = 0
    age = 0
    if len(self.prey_deaths) > 0:
      captured= 1
    for pred in self.predators:
      left_energy += pred.energy
      dist = self.preys[0].loc - pred.loc 
      distance += np.linalg.norm(dist)
      age += pred.age
    left_energy /= self.num_predator
    distance /= self.num_predator
    age /= self.num_predator
    self.file_fp.write(str(captured) + ' ,' + str(left_energy)+' ,'+str(distance) + ' ,' + str(age)+'\n')

  def update_cotarget(self):
    if len(self.preys) == 0:
      self.cotarget_idx = -1
      return
    target_candidates = []
    dist_sum_candidates = []
    for pred in self.predators:
      distSum = 0
      idx = pred.closest_idx
      if idx != -1:
        for pred in self.predators:
          dist = pred.loc - self.preys[idx].loc
          d = np.linalg.norm(dist)
          distSum += d
      if distSum != 0:
        target_candidates.append(idx)
        dist_sum_candidates.append(distSum)
    if len(target_candidates) == 0:
     return
    list_idx = np.argsort(dist_sum_candidates)
    target_idx = list_idx[0]
    count = 0
    for pred in self.predators:
      if pred.target_idx != target_idx:
        pred.target_idx = target_idx
        print "Predator "+ str(count) + " has new target : Prey "+str(target_idx)
      count+=1
    return 

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
  radius = 7 # 7'2"
  def __init__(self, x, y):

    self.loc = np.array([float(x), float(y)])
    self.vel = np.array([0., 0.])
    self.acc = np.array([0., 0.])
    self.maxForce = 3 # 40mph
    self.mass = 10 # 723.1lb
    self.repelRadius = 100  
    self.status = 0
  
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
        if d != 0:
          repelVec /= d*0.01
        self.applyF(repelVec)

  def preyFleeForce(self, preys):
    self.avoidForce(preys)
    self.approachForce(preys)
    self.alignForce(preys)

# Animats     
class Predator:
  radius = 9 # 9'2"
  def __init__(self, x, y, generation):
    #for testing
    #self.width = 1000
    #self.height = 700
    #-----------end of testing
    self.timeframe = 0
    self.age = 0 # how long does it live
    self.generation = generation

    #position
    self.loc = np.array([float(x), float(y)])
    self.prevLoc = np.array([float(x), float(y)])
    # velocity
    self.vel = 0.0
    self.acc = 0.0
    self.direction = Direction.N
    self.speed = Speed.up
    self.maxSpeed = 50/scale # 49.7 mph
    
    self.mass = 440 # 441lb
    self.captureRadius = 10

    #for finding target
    self.target_idx = -1
    self.closest_idx = -1

    #set default engery
    self.energy = Default_Engery 
    self.targetPrey = None

    self.touching = None
    self.behavior = Behavior.stay
    self.state = 0

    # neural net
    '''
    self.net = FeedForwardNetwork()
    self.net.addInputModule(LinearLayer(9, name='in'))
    self.net.addModule(SigmoidLayer(9, name='hidden'))
    self.net.addOutputModule(LinearLayer(3, name='out'))
    self.net.addConnection(FullConnection(self.net['in'], self.net['hidden']))
    self.net.addConnection(FullConnection(self.net['hidden'], self.net['out']))
    self.net.sortModules()
    '''
    # thresholds for deciding an action
    self.move_threshold = 0

  def getNNWInputList(self):
    nnlist = []
    nnlist.append(np.linalg.norm(self.vel))
    nnlist.append(self.energy)
    nnlist.append(self.state)
    return nnlist

  def update(self, predators, preys, info):
    #self.PredForce( preys, predators )
    #self.vel += self.acc
    #self.loc += self.vel
    #self.acc = np.array([0., 0.])
    self.direction = info[1]
    self.speed = info[0]

    # Update Direction
    orientation = np.array([0., 0.])
    if self.direction == Direction.N:
      orientation = np.array([0., -1.0])
    elif self.direction == Direction.NE:
      orientation = np.array([1.0, -1.0])
    elif self.direction == Direction.E:
      orientation = np.array([1.0, 0.])
    elif self.direction == Direction.SE:
      orientation = np.array([1.0, 1.0])
    elif self.direction == Direction.S:
      orientation = np.array([0., 1.0])
    elif self.direction == Direction.SW:
      orientation = np.array([-1.0, 1.0])
    elif self.direction == Direction.W:
      orientation = np.array([-1.0, 0.])
    elif self.direction == Direction.NW:
      orientation = np.array([-1.0, -1.0])
    orientation = normalize(orientation)

    # Update Acc
    if self.speed == Speed.up:
      self.acc = 10/scale
    elif self.speed == Speed.down:
      self.acc = -20/scale
    elif self.speed == Speed.maintain:
      self.acc = 0

    # Update Speed
    deltaVel = self.vel
    self.vel += self.acc
    if self.vel >= self.maxSpeed:
      self.vel = self.maxSpeed
    elif self.vel <= 0:
      self.vel = 0.0
    deltaVel = abs(self.vel - deltaVel)
    #print self.vel

    # Update Location
    self.prevLoc = self.loc
    self.loc += orientation*self.vel

    # Update Energy
    self.consumeEnergy(self.vel)

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

    #sensors = ()
    '''decision = self.net.activate(sensors)'''
    # consume energy based on differnt current behavior
    self.age += 1
    self.timeframe += 1

  def consumeEnergy(self, v):
    self.energy -= v

  # returns a child with a genetic combination of neural net weights of 2 parents
  def mate(self, other):
    child = Predator(0,0)
    child.generation = min(self.generation, other.generation) + 1
    # inherit parents connection weights
    for i in range(0,len(self.net.params)):
      if random.random() > .05:
         child.net.params[i] = random.choice([self.net.params[i], other.net.params[i]])
    return child

  '''
  def applyF(self, force):
    # F = ma (a = F/m)
    a = force / self.mass
    self.acc += a

  def approachForce(self, preys):
    count = 0
    approachRadius = self.mass + 260
#    captureRadius = 3.0
    min_dist = sys.float_info.max
    min_idx = -1
    #find the closest prey
    for prey in preys:
      dist = prey.loc - self.loc
      d = np.linalg.norm(dist)

      #if d < min_dist and d < approachRadius:
      if d < min_dist:
        min_idx = count
      count+=1
    if min_idx != -1:
      self.closest_idx = min_idx
    #set it at the first time, still -1 if no prey around
    if self.target_idx == -1:
      self.target_idx = self.closest_idx

    #approach the closest prey
    if self.target_idx != -1 and len(preys) > 0: 
      approachVec = preys[self.target_idx].loc - self.loc
      approachVec = normalize(approachVec)
      approachVec *= self.maxForce*2
      #approachVec = self.updateForce(approachVec)
      #approachVec = self.updateEngery(approachVec)
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
  '''

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
  '''
  def PredForce(self, preys, prads):
    self.approachForce(preys)
    #self.avoidForce(prads)
  '''