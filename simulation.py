#!/usr/bin/python
import animats
import sys  # sys.exit()
import pygame
import math
import os
from readFile import *
import svm_learn
import NNW
import time
import datetime
import readFile
import random

class Simulation:
  def __init__(self, generation, num_preds, num_preys, width, height, saved_nets, speed_para, dir_para):
    # initialize pygame
    pygame.init()

    # initialize the screen
    self.size = width, height
    self.screen = pygame.display.set_mode(self.size)
    self.screenWidth = width
    self.screenHeight = height

    # set the name of display windows
    pygame.display.set_caption('Import/Export project')

    #initialize sprites
    self.bg = pygame.image.load("resources/bg.png")
    
    # pictures resources
    self.pred_sprite    = pygame.image.load("resources/pred.png")
    self.prey           = pygame.image.load("resources/prey.png")

    # modify pictures to appropriate sizes
    self.pred_sprite   = pygame.transform.scale(self.pred_sprite, (18,18))
    self.bg            = pygame.transform.scale(self.bg, (1000, 700))
    self.prey          = pygame.transform.scale(self.prey, (18, 18))

    self.env = animats.Environment(generation, num_preds, num_preys, width, height, saved_nets)
    self.env.set_nn_para(speed_para,dir_para)

  def end(self):
    return self.env.end_iteration()

  def texts(self, score, line, font_size):
    # font=pygame.font.Font(None,font_size)
    # text=font.render(str(score), 1,(0,0,0))
    # self.screen.blit(text, (30, 30+int(line)*15)) 
    print score

  def update(self, speed):
    # update model a certain number of times
    for i in range(speed):
      self.env.update()
    # repaint
    self.screen.blit(self.bg, (0,0))
    # paint prey
    for prey in self.env.preys:
      self.screen.blit(self.prey, (prey.loc[0] - prey.radius, prey.loc[1] - prey.radius))
    # paint predator
    for pred in self.env.predators:
      self.screen.blit(pygame.transform.rotate(self.pred_sprite, 360), (pred.loc[0] - pred.radius, pred.loc[1] - pred.radius))
    # paint text
    count = 0
    for pred in self.env.predators:
      # self.texts("PREDATOR " + str(count+1), count*6, 20)
      # self.texts("  state: " + pred.speed_text, count*6 + 1, 20)
      # self.texts("  direction: " + pred.direction_text, count*6 + 2, 20)
      # self.texts("  speed: " + str(pred.vel*5), count*6 + 3, 20)
      # self.texts("  energy: " + str(pred.energy), count*6 + 4, 20)
      count += 1
    pygame.display.flip()
#--- End of Simulation Class --- #

def get_last_line(file):
  f = open(file,'r')
  for line in f:
    pass
  return line

# train_list have the path of all the training data
def readTrainData(train_list, Train_Num, top_num):
  list_len = len(train_list)
  InputSamples = []
  SpeedSamples = []
  DirectionSamples = []
  # Load each sample from train files
  randomChoose = random.sample( range(5),  top_num)
  print randomChoose
  for i in randomChoose:
    f = open(train_list[i][1],'r')
    if f.closed:
      continue
    for row in f:
      row = row.replace("\n", "")
      data = row.split(',')
      InputSamples.append(tuple(data[0:30]))
      SpeedSamples.append(tuple(data[30:39]))
      DirectionSamples.append(tuple(data[39:63]))
    f.close()

  return InputSamples, SpeedSamples, DirectionSamples 

def TrainPrevGen(train_list, Train_Num, top_num):
  Init_Speed_Net = NNW.NNW(30,42,9)
  Init_Dir_Net = NNW.NNW(30,42,24)
  
  InputSamples, SpeedSamples, DirectionSamples = readTrainData(train_list, Train_Num, top_num)

  time_start = datetime.datetime.now()
  # Train the data
  print '# In TrainPrevGen Function :'
  Init_Speed_Net.setTrainData(InputSamples,SpeedSamples)
  Init_Dir_Net.setTrainData(InputSamples,DirectionSamples)
  print '- Start Train Best-' + str(top_num) + ' Data :' + str(time_start)
  Init_Speed_Net.trainData()
  Init_Dir_Net.trainData()
  time_finish = datetime.datetime.now()
  delta_time = time_finish - time_start
  print '- End of Training -- Training Time: ' + str(delta_time )
  # Get parameters of two Neural Network
  Speed_Parameter = Init_Speed_Net.getParameter()
  Dir_Parameter   = Init_Dir_Net.getParameter()
  del Init_Dir_Net
  del Init_Speed_Net
  del InputSamples
  del SpeedSamples
  del DirectionSamples
  return Speed_Parameter, Dir_Parameter

def UpdateFitness(fitness_list, filename, max_num):
  file_last_line = get_last_line(filename)
  file_last_line = file_last_line.replace('\n','')
  file_last_line = file_last_line.split(',')
  iter_fitness_val = file_last_line[-1]

  if len(fitness_list) > 0:
    lowest_fitness = fitness_list[0][0]

  # append file if number is less then the number we want to train
  if len(fitness_list) < max_num:
    fitness_list.append((iter_fitness_val, filename))
    fitness_list.sort(lambda x,y:cmp(x[0],y[0]))
  # add the file into fitness_list if the fitness value is better than the lowest one
  elif float(lowest_fitness) < float(iter_fitness_val):
    fitness_list.pop(0)
    fitness_list.append((iter_fitness_val, filename))
    fitness_list.sort(lambda x,y:cmp(x[0],y[0]))
  return  fitness_list


def main():
  # Read Successful Prey-Capture Data and generate trained parameters ---#
  # read from a folder of sample data created by force-predator
  (InputSamples, SpeedSamples, DirectionSamples, Fitness) = ReadSampleData("sampleData")
  InputSamples, SpeedSamples, DirectionSamples = readTrainData(Fitness, 5, 2)

  Init_Speed_Net = NNW.NNW(30,42,9)
  Init_Dir_Net = NNW.NNW(30,42,24)
  Init_Speed_Net.setTrainData(InputSamples,SpeedSamples)
  Init_Dir_Net.setTrainData(InputSamples,DirectionSamples)
  Init_Speed_Net.trainData()
  Init_Dir_Net.trainData()
  Speed_Parameter = Init_Speed_Net.getParameter()
  Dir_Parameter   = Init_Dir_Net.getParameter()
  del Init_Dir_Net
  del Init_Speed_Net
  # ---- End of generating trained parameters ---------------------------#

  # ---- Initial Variables -----------#
  Generation = 0
  Iteration  = 0
  Max_Gen    = 1
  Max_Iter   = 1
  Train_Num  = 15
  Top_train_num = 2

  Predator_Num = 3
  Prey_Num = 1
  Window_Width = 1000
  Window_Height = 700
  # ---- End of Initial Variables ----#

  # ---- Simulation Usage ---- #
  if len(sys.argv) < 3:
    print "Usage   : python simulation.py [gen number] [iter number per gen]"
    print "Example : python simulation.py 3 2"
  '''
  if len(sys.argv) >= 4:
    Predator_Num = int(sys.argv[3])
  if len(sys.argv) >= 5:
    Prey_Num = int(sys.argv[4])
  '''
  Max_Gen  = int(sys.argv[1])
  Max_Iter = int(sys.argv[2])

  # --- Main Loop --- #
  while Generation < Max_Gen:
    Iteration = 0
    print '# In Main Loop :'
    print '-- Start Generation ' + str(Generation) + '  --'
    if Generation!= 0:
      Speed_Parameter, Dir_Parameter = TrainPrevGen(Fitness, Train_Num, Top_train_num)
    # --- Iteration Loop --- #
    while Iteration < Max_Iter:
      FilePath = 'Gen_Data/data_sample_gen_' + str(Generation) + '_iter_' + str(Iteration) + '.csv'
      simulation = Simulation(Generation, Predator_Num, Prey_Num, Window_Width, Window_Height, FilePath, Speed_Parameter, Dir_Parameter)
      while(simulation.end()!=1):
        #----- Check for exit ------#
        for event in pygame.event.get():
          if event.type == pygame.QUIT: 
            sys.exit()
        #----- End of Check --------#
        simulation.update(1)
      Fitness = UpdateFitness(Fitness,FilePath,Train_Num)
      Iteration += 1
    # --- End of Iteration ---#
    print '-- End of Generation ' + str(Generation) + ' --'
    Generation += 1
  # --- End of Loop --- #

if __name__ == '__main__':
  main()
