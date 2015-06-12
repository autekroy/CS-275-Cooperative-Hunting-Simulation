#!/usr/bin/python
import pickle
import random
import math
from enum import Enum
import numpy as np
import sys
import NNW
# import Prey
# import Predator_Force as Predator
import Predator
import Prey_simple as SPrey
import simulation
import os

from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

preyFleeing = 0
Default_Engery = 9000
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

def getFit(got_pray,energy,dist, age):
    return 1000000 * got_pray + energy + 1000/dist + 0.1*age

class Environment:
  def __init__(self, generation, num_predator, num_prey, width, height, filename):

    # environment
    self.width = width
    self.height = height
    self.num_predator = num_predator
    self.num_prey = num_prey

    # record log
    self.log = []
    self.moveLog = []

    # record file
    self.filename = filename
    self.file_fp = None
    if filename!='':
      self.file_fp = open(filename,'w')

    self.timeframe = 0
    self.cotarget_idx = None
    self.pred_deaths = []
    self.predators = []
    self.capturedPrey = []
    self.placeRadius = 200;
    self.halt = 0

    #saved_states = self.load()

    # Initial prey instances
    self.preys = []
    self.prey_deaths = []
    for i in range(self.num_prey):
      p = SPrey.Prey_simple(400+random.random() * 200, 250+random.random() * 200)
      self.preys.append(p)

    # Initial predator instances
    for i in range(self.num_predator):
      pos = self.findSpace(i, 200, 20, Predator.Predator.radius)
      #if len(saved_states) > 0:
      #  a = saved_states.pop(0)
      #  a.x = pos[0]
      #  a.y = pos[1]
      #else:
      a = Predator.Predator(pos[0], pos[1], generation)
      #a.generation = 1
      self.predators.append(a)
      
    #---------Neural Network----------#
    #-- Centrol Controller -----------#
    #-- First Network ---: Speed -----#
    self.speed_net = NNW.NNW(30,42,9)
    #-- Second Network --: Direction -#
    self.dir_net = NNW.NNW(30,42,24)
    #---------------------------------#
    # End of Initial Class

  # Set the trained weights for the NN
  def set_nn_para(self, speedPara, dirPara):
    self.speed_net.setParameters(speedPara)
    self.dir_net.setParameters(dirPara)

  # Find if End of the iteration
  def end_iteration(self):
    return self.halt

  # Get the Input Node's values
  def getNNWInput(self):
    input_vals = []
    for pred in self.predators:
      input_vals = input_vals + pred.getNNWInputList()
      for other in self.predators:
        if pred != other:
          loc = other.loc - pred.loc
          loc = normalize(loc)
          input_vals.append(loc[0])
          input_vals.append(loc[1])
      loc = self.preys[0].loc - pred.loc
      loc = normalize(loc)
      input_vals.append(loc[0])
      input_vals.append(loc[1])
    input_vals = input_vals + self.preys[0].getNNInput()
    return input_vals

  # Locate the coordinates for one animat
  def findSpace(self, count, placeRadius, noCoverDegree, AnimateRadius):
    degree = random.randrange(noCoverDegree , 360.0/self.num_predator - noCoverDegree)  # random degree
    degree = degree + count * 360.0/self.num_predator
    degree = math.radians(degree) 
    #Convert angle from degrees to radians.
    radius = random.randrange(placeRadius, placeRadius + 20)
    x = math.cos(degree) * radius
    y = math.sin(degree) * radius
    centerX = self.width / 2
    centerY = self.height /2
    x = centerX + x
    y = centerY + y
    return (x, y)

  # Derive info(speed_status,direction) from Output Nodes of two NNs
  def extract_info_from_nnlist(self, nnlist_speed, nnlist_dir):
    info = []
    n_speed_idx_list = np.argsort(nnlist_speed)
    n_speed_top = n_speed_idx_list[::-1][0]
    if n_speed_top == 0:
      info.append(0)
    elif n_speed_top == -1:
      info.append(-1)
    else:
      info.append(1)
    n_dir_idx_list = np.argsort(nnlist_dir)
    n_dir_top = n_dir_idx_list[::-1][0]
    info.append(n_dir_top)
    return info

  # Filtering the Output of two NNs
  def filt_with_threshold(self, l1, l2, ans):
    for i in range(0, self.num_predator):
      for j in range(0,len(l1)/self.num_predator):
        if j == ans[i][0]:
          l1[i*3+j] = 1.0
        else:
          l1[i*3+j] = 0.0
      for j in range(0,len(l2)/self.num_predator):
        if j == ans[i][1]:
          l2[i*8+j] = 1.0
        else:
          l2[i*8+j] = 0.0
    return list(l1)+list(l2)

  # Formulate the Recording Format of Inputs, Outputs, and Fitness infos
  def get_record_list(self, out1, out2, ins, ans):
    return list(ins) + list(self.filt_with_threshold(out1,out2,ans))


  # Check if any animat is dead
  def check_if_dead(self):
    while len(self.pred_deaths ) > 0:
      self.predators.remove(self.pred_deaths.pop(0))
      print "Predator Dies"
    while len(self.prey_deaths ) > 0:
      self.preys.remove(self.prey_deaths.pop(0))
      print "Prey Captured"

  # Update the whole simulated Animats
  def update(self):
    # check if dead
    self.check_if_dead()
    # update each prey
    for prey in self.preys:
      prey.update(self.preys, self.predators)
    # get the result from NNW
    input_vals = self.getNNWInput()
    nn_out_speed = self.speed_net.activate(input_vals)
    nn_out_dir = self.dir_net.activate(input_vals)

    count = 0
    # Record variable for the desicions made by two NNs
    cur_res = []
    # update each predator
    for pred in self.predators:
      update_info = self.extract_info_from_nnlist(nn_out_speed[count*3:count*3+3],nn_out_dir[count*8:count*8+8])
      # Update Speed_status and Direction
      deadPrey = pred.update(self.predators, self.preys, update_info)
      # Check if any prey is captured
      if (deadPrey != None) and (deadPrey not in self.prey_deaths):
        self.prey_deaths.append(deadPrey)
      # Check if the predator is dead
      if pred not in self.pred_deaths and (pred.energy <= 0):
        self.pred_deaths.append(pred)
      count += 1
      cur_res.append(update_info)

    # End Iteration if animats found dead
    if len(self.pred_deaths) > 0 or len(self.prey_deaths) > 0:
      self.halt = 1
      #Record Fitness
      self.record(self.get_record_list(nn_out_speed,nn_out_dir,input_vals, cur_res))
      self.file_fp.close()
      return

    # Update the cooperative target
    self.update_cotarget()
    self.timeframe += 1
    # Record Sample
    if self.timeframe >= 1:
      self.record(self.get_record_list(nn_out_speed,nn_out_dir,input_vals, cur_res))
      self.timeframe = 0  
    return 

  # The fullowing function record  one input nodes, two output nodes, 5 infos
  # For input node: 3 x ( predator: norm(velocity), norm(energy), state,
  #                                 norm(vector(other_predator,predator)) x2,
  #                                 norm(vector(prey, predator))
  #                 1 x (prey: state, norm(current_direction))
  # For output node 1: 3 x ( predator: direction)
  # For output node 2: 3 x ( predator: speed_status)
  # append (captured, left_engery, dis, age, fitness) at the end
  def record(self, r_list):
    if self.file_fp == None:
      return
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
    fit = getFit(captured, left_energy, distance, age)
    self.file_fp.write(str(captured) + ' ,' + str(left_energy)+' ,'+str(distance) + ' ,' + str(age)+' ,'+str(fit)+'\n')

  # Update the Cooperative hunting targets
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

  # Find is two animtas Collid with each other ------ Not in Used
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

  # load animat states ----- Not in Used
  '''
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
  '''
  # save neural net states ---- Not in Used
  '''
  def save(self):
    if self.filename != "":
      f = open(self.filename, 'w')
      pickle.dump(self.predators, f)
      f.close()
  '''
  # Get the radius of prey
  def getPrey_Radius():
    return SPrey.radius() 


   
