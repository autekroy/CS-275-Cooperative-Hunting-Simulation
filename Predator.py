import random
import math
from enum import Enum
import numpy as np

scale = 5.0
Default_Engery = 1000
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
  
class Predator:
  radius = 9
  def __init__(self, x, y, generation):
    #self.radius = 9 # 9'2"
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
    self.vel = 1.0
    self.acc = 0.1
    self.direction = Direction.N
    self.speed = Speed.up
    self.maxSpeed = 50/scale # 49.7 mph
    
    self.mass = 440 # 441lb
    self.captureRadius = 10

    #for finding target
    self.target_idx = 0
    self.closest_idx = 0

    #set default engery
    self.energy = Default_Engery 
    self.targetPrey = None

    self.touching = None
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
    #print str(self.state) + ' : ' + str(self.vel)
    #self.PredForce( preys, predators )
    #self.vel += self.acc
    #self.loc += self.vel
    #self.acc = np.array([0., 0.])
    self.direction = info[1]
    self.speed = info[0]
    self.state = self.speed
    # Update Direction
    orientation = np.array([0., 0.])
    if self.direction == 0:
      orientation = np.array([0., -1.0])
    elif self.direction == 1:
      orientation = np.array([1.0, -1.0])
    elif self.direction == 2:
      orientation = np.array([1.0, 0.])
    elif self.direction == 3:
      orientation = np.array([1.0, 1.0])
    elif self.direction == 4:
      orientation = np.array([0., 1.0])
    elif self.direction == 5:
      orientation = np.array([-1.0, 1.0])
    elif self.direction == 6:
      orientation = np.array([-1.0, 0.])
    elif self.direction == 7:
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
    deltaVel = (self.vel + deltaVel)/2
    #print self.vel

    # Update Location
    self.prevLoc = self.loc
    self.loc += orientation*self.vel

    # Update Energy
    self.consumeEnergy(deltaVel)

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