#!/usr/bin/python
import pickle
import random
import math
from enum import Enum
import numpy as np
import sys
import NNW
import Prey
import Predator
import Prey_simple as SPrey

from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

preyFleeing = 0
Default_Engery = 1000
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
      p = SPrey.Prey_simple(400+random.random() * 200, 250+random.random() * 200)
      self.preys.append(p)

    # create predator instances
    for i in range(self.num_predator):
      # print i
      pos = self.findSpace(i, 200, 20, Predator.Predator.radius)
      if len(saved_states) > 0:
        a = saved_states.pop(0)
        a.x = pos[0]
        a.y = pos[1]
      else:
        a = Predator.Predator(pos[0], pos[1], generation)
        a.generation = 1
      self.predators.append(a)
    #---------Neural Network----------#
    #-- Method 1 ---------------------#
    #-- Initial Stage ----------------#
    #-- First Network ---: Speed -----#
    self.speed_net = NNW.NNW(28,24,9)
    #-- Second Network --: Direction -#
    self.dir_net = NNW.NNW(28,38,24)
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
      info.append(0)
    elif n_speed_top == -1:
      info.append(-1)
    else:
      info.append(1)
    n_dir_idx_list = np.argsort(nnlist_dir)
    n_dir_top = n_dir_idx_list[::-1][0]
    info.append(n_dir_top)
    return info

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
    return l1,l2

  def update(self):
    # check if dead
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
    cur_res = []
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
      cur_res.append(update_info)
 
    if len(self.pred_deaths) > 0 or len(self.prey_deaths) > 0:
      self.halt = 1
      nn_out_speed, nn_out_dir = self.filt_with_threshold(nn_out_speed,nn_out_dir,cur_res)
      r_list = list(input_vals) + list(nn_out_speed) + list(nn_out_dir)
      self.record(r_list)
      self.file_fp.close()
      return

    nn_out_speed, nn_out_dir = self.filt_with_threshold(nn_out_speed,nn_out_dir,cur_res)
    r_list = list(input_vals) + list(nn_out_speed) + list(nn_out_dir)
    self.update_cotarget()
    self.timeframe += 1
    if self.timeframe >= 5:
      self.record(r_list)
      self.timeframe = 0  

  # The fullowing function record input nodes 
  # two output nodes
  # append captured, left_engery, dis, age at the end
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
  def getPrey_Radius():
    return SPrey.radius()



   