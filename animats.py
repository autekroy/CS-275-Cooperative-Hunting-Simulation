#!/usr/bin/python
import pickle
import random
import math
import numpy as np
from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

preyFleeing = 0

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
    self.num_animats = num_animats
    self.deaths = []
    self.animats = []
    saved_states = self.load()


    # prey
    self.preys = []
    num_preys = 5
    for i in range(num_preys):
      p = Prey(400+random.random() * 200, 250+random.random() * 200)
      self.preys.append(p)

    while len(self.animats) < num_animats:
      pos = self.findSpace(Animat.radius, (0, self.height))
      if len(saved_states) > 0:
	a = saved_states.pop(0)
	a.x = pos[0]
	a.y = pos[1]
      else:
	a = Animat(pos[0], pos[1], random.random() * 360)
	a.generation = 1
      self.animats.append(a)
  # prey

  # line of sight
  def line_of_sight(self, animat):
    step_x = int(math.cos(animat.direction*math.pi / 180) * 10)
    step_y = int(math.sin(animat.direction*math.pi / 180) * 10)
    new_x = animat.x + step_x
    new_y = animat.y + step_y
    sees = None
    while not sees:
      new_x += step_x
      new_y += step_y
      sees = self.collision(new_x, new_y, Animat.radius, animat)
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
      pos = self.findSpace(Animat.radius, (0, self.height))
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
      prey.update(self.preys, self.animats)

    # update each animat
    for animat in self.animats:
      # Sight
      animat.sees = self.line_of_sight(animat)
      # Touch
      step = 3
      step_x = int(math.cos(animat.direction*math.pi / 180) * step)
      step_y = int(math.sin(animat.direction*math.pi / 180) * step)
      animat.touching = self.collision(animat.x + step_x, animat.y + step_y, Animat.radius, animat)
      # update
      animat.update()
      # moving
      if animat.wants_to_move and \
	(not animat.touching or isinstance(animat.touching,Food)):
	animat.x = step_x + animat.x
	animat.y = step_y + animat.y

      # pickup
      if isinstance(animat.touching, Food) and animat.wants_to_pickup:
	self.foods.remove(animat.touching)
        animat.food = animat.touching
      # putdown
      if animat.wants_to_putdown:
	if isinstance(animat.food, Fruit):
	  self.foods.append(Fruit(animat.x - (step_x*10), animat.y - (step_y*10)))
	elif isinstance(animat.food, Veggie):
	  self.foods.append(Veggie(animat.x - (step_x*10), animat.y - (step_y*10)))
	animat.food = None
      # keep the food supply constant
      # self.produceFoods()
      # DEATH 
      if animat not in self.deaths \
      and (animat.fruit_hunger + animat.veggie_hunger < 1000):
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
      if (x - animat.x)**2 + (y - animat.y)**2 <= Animat.radius**2:
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

    self.loc = [float(x), float(y)]
    self.vel = [0., 0.]
    self.acc = [0., 0.]
    self.maxForce = 3
    self.mass = 10   
  
  def update(self, preys, preds):
    self.repelForce(preds, 50)
    if preyFleeing == 1:
      self.preyForce(preys)
      self.vel = np.add(self.vel, self.acc)
      self.loc = np.add(self.loc, self.vel)      
    else:
      self.vel = [random.random(), random.random()]
      self.loc = np.add(self.loc, self.vel)
    self.acc = [0., 0.]

  def applyF(self, force):
    force /= self.mass
    self.acc = np.add(self.acc, force)

  def avoidForce(self, preys):
    count = 0
    locSum = [0., 0.]
    for p in preys:
      separation = self.mass + 20
      dist = np.subtract(p.loc, self.loc)
      d = np.linalg.norm(dist)
      if d != 0 and d < separation:
        locSum = np.add(locSum, p.loc)
        count += 1
    if count > 0:
      locSum /= count
      avoidVec = np.subtract(self.loc, locSum)
      avoidVec = limit(avoidVec, self.maxForce*2.5)
      self.applyF(avoidVec)

  def approachForce(self, preys):
    count = 0
    locSum = [0., 0.]
    for p in preys:
      approachRadius = self.mass + 60
      dist = np.subtract(p.loc, self.loc)
      d = np.linalg.norm(dist)
      if d != 0 and d < approachRadius:
        locSum = np.add(locSum, p.loc)
        count += 1
    if count > 0:
      locSum /= count
      approachVec = np.subtract(locSum, self.loc)
      approachVec = limit(approachVec, self.maxForce)
      self.applyF(approachVec)

  def alignForce(self, preys):
    count = 0
    velSum = [0., 0.]
    for p in preys:
      alignRadius = self.mass + 100
      dist = np.subtract(p.loc, self.loc)
      d = np.linalg.norm(dist)
      if d != 0 and d < alignRadius:
        velSum = np.add(velSum, p.vel)
        count += 1
      if count > 0:
        velSum /= count
        alignVec = velSum
        alignVec = limit(alignVec, self.maxForce)
        self.applyF(alignVec)

  def repelForce(self, preds, r):
    for pred in preds:
      pred_loc = [pred.x, pred.y]
      futurePos = np.add(self.loc, self.vel)
      dist = np.subtract(pred_loc, futurePos)
      d = np.linalg.norm(dist)

      if d <= r:
        global preyFleeing
        if preyFleeing == 0:
          preyFleeing = 1
        repelVec = np.subtract(self.loc, pred_loc)
        normalize(repelVec)
        repelVec = np.multiply(repelVec, self.maxForce*5)
        self.applyF(repelVec)

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
class Animat:
  radius = 30

  def __init__(self, x, y, direction):
    self.age = 0
    # position
    self.x = x
    self.y = y
    # number of going back and forth for different foods
    self.backForth = 0
    self.LastFood = None # the last food animat ate
    # orientation (0 - 359 degrees)
    self.direction = direction
    # carrying food
    self.food = None
    # touching anything
    self.touching = None
    self.sees = None
    # hunger sensor
    self.fruit_hunger = 2000
    self.veggie_hunger = 2000
    self.avg_fruit_hunger = 0
    self.avg_veggie_hunger = 0
    # neural net
    self.net = FeedForwardNetwork()
    self.net.addInputModule(LinearLayer(12, name='in'))
    self.net.addModule(SigmoidLayer(13, name='hidden'))
    self.net.addOutputModule(LinearLayer(6, name='out'))
    self.net.addConnection(FullConnection(self.net['in'], self.net['hidden']))
    self.net.addConnection(FullConnection(self.net['hidden'], self.net['out']))
    self.net.sortModules()
    # thresholds for deciding an action
    self.move_threshold = 0
    self.pickup_threshold = 0
    self.putdown_threshold = 0
    self.eat_threshold = 0
    
  def update(self):
    sensors = (2000*int(isinstance(self.sees, Fruit) or \
		        (isinstance(self.sees, Animat) and \
	                 isinstance(self.sees.food, Fruit))),
	       2000*int(isinstance(self.sees, Veggie) or \
	                (isinstance(self.sees, Animat) and \
		         isinstance(self.sees.food, Veggie))),
	       2000*int(isinstance(self.sees, Animat)),
	       2000*int(isinstance(self.sees, Environment)),
	       2000*int(isinstance(self.food, Fruit)),
	       2000*int(isinstance(self.food, Veggie)),
	       self.fruit_hunger,
	       self.veggie_hunger,
	       2000*int(isinstance(self.touching, Fruit) or \
		        (isinstance(self.touching, Animat) and \
		         isinstance(self.touching.food, Fruit))),
	       2000*int(isinstance(self.touching, Veggie) or \
		        (isinstance(self.touching, Animat) and \
		         isinstance(self.touching.food, Veggie))),
	       2000*int(isinstance(self.touching, Animat)),
	       2000*int(isinstance(self.touching, Environment)))
    decision = self.net.activate(sensors)
    # get a little hungry no matter what
    self.age += 1
    self.get_hungry(.5)
    # move forward
    self.wants_to_move = (decision[0] > self.move_threshold)
    # rotate left 
    self.direction -= decision[1]
    # rotate right 
    self.direction += decision[2]

    # pickup
    self.wants_to_pickup = ((decision[3] > self.pickup_threshold) 
			    and not self.food)
    # putdown
    self.wants_to_putdown = ((decision[4] > self.putdown_threshold)
			     and self.food)
    # eat
    if (decision[5] > self.eat_threshold) and self.food:
      if isinstance(self.food, Fruit):
	self.fruit_hunger = 2000 if (self.fruit_hunger > 1800) else (self.fruit_hunger + 200)
        self.avg_fruit_hunger = (self.avg_fruit_hunger + self.fruit_hunger) / 2
	if isinstance(self.LastFood, Veggie): # the last food is different from eating food
          self.backForth = self.backForth + 1
          # print self.backForth
        self.LastFood = Fruit(0, 0)
      elif isinstance(self.food, Veggie):
        self.veggie_hunger = 2000 if (self.veggie_hunger > 1800) else (self.veggie_hunger + 200)
        self.avg_veggie_hunger = (self.avg_veggie_hunger + self.veggie_hunger) / 2
	if isinstance(self.LastFood, Fruit):
          self.backForth = self.backForth + 1
          # print self.backForth
        self.LastFood = Veggie(0, 0)
      self.food = None
      
  def get_hungry(self, amount):
    self.fruit_hunger -= amount
    self.veggie_hunger -= amount

  # returns a child with a genetic combination of neural net weights of 2 parents
  def mate(self, other):
    child = Animat(0,0, random.random() * 360)
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
