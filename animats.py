#!/usr/bin/python
import pickle
import random
import math
from enum import Enum
import numpy as np

from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
Default_Engery = 50000

class Behavior(Enum): 
    stay = 0
    stalk  = 1
    hunt = 2



class Environment:
  def __init__(self, num_animats, width, height, filename):
    # training mode (foods everywhere)
    self.training_mode = False
    # environment
    self.width = width
    self.height = height
    # record log
    self.log = []
    self.moveLog = []
    # save state
    self.filename = filename
    # foods
    self.num_foods = num_animats
    self.foods = []
    # self.produceFoods
    # animats
    self.num_animats = 3
    self.deaths = []
    self.animats = []
    saved_states = self.load()


    # prey
    self.preys = []
    num_preys = 30
    for i in range(num_preys):
      p = Prey(random.random() * 360, random.random() * 360)
      self.preys.append(p)

    while len(self.animats) < num_animats:
      pos = self.findSpace(Predator.radius, (0, self.height))
      if len(saved_states) > 0:
        a = saved_states.pop(0)
        a.x = pos[0]
        a.y = pos[1]
      else:
        a = Predator(pos[0], pos[1])
        a.generation = 1
      self.animats.append(a)
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

  def findSpace(self, radius, bounds):
    spawns_x = range(0, self.width, 10)
    spawns_y = range(bounds[0], bounds[1], 10)
    random.shuffle(spawns_x)
    random.shuffle(spawns_y)
    for x in spawns_x:
      for y in spawns_y:
	if not self.collision(x, y, radius):
	  return (x, y)


  def update(self):
    # if an animat died, the two fittest animats mate
    while len(self.deaths) > 0: 
      fittest = sorted(self.animats, key=lambda a: -a.avg_fruit_hunger -a.avg_veggie_hunger)
      pos = self.findSpace(predator.radius, (0, self.height))
      child = fittest[0].mate(fittest[1])
      child.x = pos[0]
      child.y = pos[1]
      self.animats.append(child)
      # log dead animats stats
      tmpLog = (self.deaths[0].generation, self.deaths[0].age )
      self.log.append( tmpLog )
      tmpMoveLog = (self.deaths[0].generation, self.deaths[0].backForth)
      print str(tmpLog) + "   " + str(tmpMoveLog)
      self.moveLog.append( tmpMoveLog )
      self.animats.remove(self.deaths.pop(0))
    
    # update each prey
    for prey in self.preys:
      prey.update(self.preys)

    # update each animat
    for animat in self.animats:
      # Sight
      animat.sees = self.line_of_sight(animat)
      # Touch
      step = 3
      step_x = int(math.cos(animat.direction*math.pi / 180) * step)
      step_y = int(math.sin(animat.direction*math.pi / 180) * step)
      animat.touching = self.collision(animat.loc[0] + step_x, animat.loc[1] + step_y, Predator.radius, animat)
      # update
      animat.update()
      animat.loc[0] = step_x + animat.loc[0]
      animat.loc[1] = step_y + animat.loc[1]
      # moving
 #      if animat.wants_to_move and \
	# (not animat.touching or isinstance(animat.touching,Food)):
	# animat.x = step_x + animat.x
	# animat.y = step_y + animat.y

      # pickup
 #      if isinstance(animat.touching, Food) and animat.wants_to_pickup:
	# self.foods.remove(animat.touching)
 #        animat.food = animat.touching
 #      # putdown
 #      if animat.wants_to_putdown:
	# if isinstance(animat.food, Fruit):
	#   self.foods.append(Fruit(animat.x - (step_x*10), animat.y - (step_y*10)))
	# elif isinstance(animat.food, Veggie):
	#   self.foods.append(Veggie(animat.x - (step_x*10), animat.y - (step_y*10)))
	# animat.food = None
      # keep the food supply constant
      # self.produceFoods()
      # DEATH 
      if animat not in self.deaths \
      and (animat.energy < 0):
	self.deaths.append(animat)
        

  def collision(self, x, y, radius, without=None):
    # check wall collision
    if (y + radius) > self.height or (x + radius) > self.width  \
    or (x - radius) < 0 or (y - radius) < 0:
      return self
    # check food collision
    for food in self.foods:
      if (x - food.x)**2 + (y - food.y)**2 <= Food.radius**2:
	return food
    # check animat-animat collision
    animats = list(self.animats)
    if without:
      animats.remove(without)
    for animat in animats:
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
      animats = pickle.load(f)
      f.close()
      return animats
    except:
      print "Could not load file " + self.filename
      return []

  # save neural net states
  def save(self):
    if self.filename != "":
      f = open(self.filename, 'w')
      pickle.dump(self.animats, f)
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
  
  def update(self, preys):
    self.preyForce(preys)
    self.vel = np.add(self.vel, self.acc)
    self.loc = np.add(self.loc, self.vel)
    self.acc = [0., 0.]

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
    #self.pickup_threshold = 0
    #self.putdown_threshold = 0
    #self.eat_threshold = 0
    
  def update(self):

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

    sensors = (

      )
    '''decision = self.net.activate(sensors)'''
    # get a little hungry no matter what
    #self.age += 1
    self.get_hungry(.5)
    # move forward
    #self.wants_to_move = (decision[0] > self.move_threshold)
    # rotate left 
    #self.direction -= decision[1]
    # rotate right 
    #self.direction += decision[2]

    # pickup
    #self.wants_to_pickup = ((decision[3] > self.pickup_threshold) 
			    #and not self.food)
    # putdown
    #self.wants_to_putdown = ((decision[4] > self.putdown_threshold)
			     #and self.food)
    # eat
    #if (decision[5] > self.eat_threshold) and self.food:
      #if isinstance(self.food, Fruit):
	#self.fruit_hunger = 2000 if (self.fruit_hunger > 1800) else (self.fruit_hunger + 200)
        #self.avg_fruit_hunger = (self.avg_fruit_hunger + self.fruit_hunger) / 2
	#if isinstance(self.LastFood, Veggie): # the last food is different from eating food
          #self.backForth = self.backForth + 1
          # print self.backForth
        #self.LastFood = Fruit(0, 0)
      #elif isinstance(self.food, Veggie):
        #self.veggie_hunger = 2000 if (self.veggie_hunger > 1800) else (self.veggie_hunger + 200)
        #self.avg_veggie_hunger = (self.avg_veggie_hunger + self.veggie_hunger) / 2
	#if isinstance(self.LastFood, Fruit):
          #self.backForth = self.backForth + 1
          # print self.backForth
        #self.LastFood = Veggie(0, 0)
      #self.food = None
      
  def get_hungry(self, amount):
    self.energy -= amount

  # returns a child with a genetic combination of neural net weights of 2 parents
  def mate(self, other):
    child = Predator(0,0, random.random() * 360)
    child.generation = min(self.generation, other.generation) + 1
    # inherit parents connection weights
    for i in range(0,len(self.net.params)):
      if random.random() > .05:
	child.net.params[i] = random.choice([self.net.params[i], other.net.params[i]])
    return child

# Fruits and Veggies
class Food:
  radius = 20
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.bites = 10

class Veggie(Food): pass
class Fruit(Food): pass
