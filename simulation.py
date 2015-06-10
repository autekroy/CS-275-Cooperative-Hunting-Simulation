#!/usr/bin/python
import animats
import sys  # sys.exit()
import pygame
import math
import os
from readFile import *
import svm_learn
import NNW

class Simulation:
  def __init__(self, generation, num_preds, num_preys, width, height, saved_nets):
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



  def ifend(self):
    return self.env.end_iteration()

  def texts(self, score, line, font_size):
    font=pygame.font.Font(None,font_size)
    text=font.render(str(score), 1,(0,0,0))
    self.screen.blit(text, (30, 30+int(line)*15)) 

  def set_nn_para(self, speed_para, dir_para):
    self.env.set_nn_para(speed_para,dir_para)

  def update(self, speed):
    # update model a certain number of times
    for i in range(speed):
      self.env.update()

    # for future 'pause' button, the parameter take milliseconds pause time
    # pygame.time.wait()

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
      self.texts("PREDATOR " + str(count+1), count*6, 20)
      self.texts("  state: " + pred.speed_text, count*6 + 1, 20)
      self.texts("  direction: " + pred.direction_text, count*6 + 2, 20)
      self.texts("  speed: " + str(pred.vel*5), count*6 + 3, 20)
      self.texts("  energy: " + str(pred.energy), count*6 + 4, 20)
      count += 1

    pygame.display.flip()



def get_last_line(file):
  f = open(file,'r')
  for line in f:
    pass
  return line

if __name__ == "__main__":

  sampleTrain, sampleTarget1, sampleTarget2 = readData("sample/data")

  sample_speed_net = NNW.NNW(28,24,9)
  sample_dir_net = NNW.NNW(28,38,24)

  sample_speed_net.setTrainData(sampleTrain, sampleTarget1)
  sample_dir_net.setTrainData(sampleTrain, sampleTarget2)

  sample_speed_net.trainOnce()
  sample_dir_net.trainOnce()
  speed_para = sample_speed_net.parameter()
  dir_para = sample_dir_net.parameter()
  
  # load save state from file
  fitness = []

  generation = 0
  iter_num = 0
  max_iter = 1
  filename = ""
  slct_num = 12


  i1 = 0
  while i1 < 10:
    data = get_last_line("training_data"+'_gen_'+str(0)+'_iter_'+str(i1)+'.csv').split(",")
    age = int(data[-1])
    dist = float(data[-2])
    energy = float(data[-3])
    if energy < 0.0:
      energy = 0.0
    got_pray = float(data[-4])
    fit = 1000000 * got_pray + 10 * energy + 100/dist + age
    print 'fit is :' + str(fit)
    if len(fitness)<slct_num: 
      fitness.append((iter_num,fit,generation))
      fitness.sort(lambda x,y:cmp(x[1],y[1]))
    elif fitness[0][1] < fit:
      fitness.pop(0)
      fitness.append((iter_num,fit,generation))
      fitness.sort(lambda x,y:cmp(x[1],y[1]))
    i1 += 1


  if len(sys.argv) > 2:
    filename = "training_data"
    generationsNum = int(sys.argv[1]) # generation number 
    max_iter = int(sys.argv[2])    # iteration number

  while generation < generationsNum:

    if generation == 0: 
      iter_num = 10
    else:
      iter_num = 0
    simulation = Simulation(generation, 3, 1, 1000, 700, filename+'_gen_'+str(generation)+'_iter_'+str(iter_num)+'.csv')
    simulation.set_nn_para(speed_para,dir_para)
    # main loop
    while iter_num < max_iter: 
      for event in pygame.event.get():
        # check for exit
        if event.type == pygame.QUIT: 
          simulation.env.save()
          # save record log
          fLog = open("log.txt",'w')
          map(lambda r: fLog.write( str(r) + '\n'), simulation.env.log)
          fLog.close()
          sys.exit()
          
      simulation.update(1)
      #print simulation.ifend()
      if simulation.ifend() == 1:
        data = get_last_line("training_data"+'_gen_'+str(generation)+'_iter_'+str(iter_num)+'.csv').split(",")
        age = int(data[-1])
        dist = float(data[-2])
        energy = float(data[-3])
        if energy < 0.0:
          energy = 0.0
        got_pray = float(data[-4])
        fit = 1000000 * got_pray + 10 * energy + 100/dist + age
        print 'fit is :' + str(fit)
        if len(fitness)<slct_num: 
          fitness.append((iter_num,fit,generation))
          fitness.sort(lambda x,y:cmp(x[1],y[1]))
        elif fitness[0][1] < fit:
          fitness.pop(0)
          fitness.append((iter_num,fit,generation))
          fitness.sort(lambda x,y:cmp(x[1],y[1]))
    
        iter_num += 1
        if iter_num < max_iter:
          simulation = Simulation(generation, 3, 1, 1000, 700, filename+'_gen_'+str(generation)+'_iter_'+str(iter_num)+'.csv')
          simulation.set_nn_para(speed_para,dir_para)


    
    inp = []
    sp_oup = []
    dr_oup = []

    j = 0
    while j < len(fitness):
      f = open("training_data"+'_gen_'+str(generation)+'_iter_'+str(fitness[j][0])+'.csv', "r")
      line = f.readline()
      while line:
        trn_data = line.split(",")
        inp.append([])
        sp_oup.append([])
        dr_oup.append([])
        for i in range(3):
          trn_data[i*9+0] = float(trn_data[i*9+0])/10
          trn_data[i*9+1] = float(trn_data[i*9+0])/1000
          v_len = math.sqrt(float(trn_data[i*9+3])**2 + float(trn_data[i*9+4])**2)
          trn_data[i*9+3] = float(trn_data[i*9+3]) / v_len
          trn_data[i*9+4] = float(trn_data[i*9+4]) / v_len
          v_len = math.sqrt(float(trn_data[i*9+5])**2 + float(trn_data[i*9+6])**2)
          trn_data[i*9+5] = float(trn_data[i*9+5]) / v_len
          trn_data[i*9+6] = float(trn_data[i*9+6]) / v_len
          v_len = math.sqrt(float(trn_data[i*9+7])**2 + float(trn_data[i*9+8])**2)
          trn_data[i*9+7] = float(trn_data[i*9+7]) / v_len
          trn_data[i*9+8] = float(trn_data[i*9+8]) / v_len
        k = 0
        while k < 28:
          inp[len(inp)-1].append(float(trn_data[k]))
          k += 1
        while k < 37:
          sp_oup[len(sp_oup)-1].append(float(trn_data[k]))
          k += 1
        while k< 61:
          dr_oup[len(dr_oup)-1].append(float(trn_data[k]))
          k += 1
        line = f.readline()
      j += 1

      NN_inp = [tuple(l) for l in inp]
      NN_sp_oup = [tuple(l) for l in sp_oup]
      NN_dr_oup = [tuple(l) for l in dr_oup]

      sp_nnw = NNW.NNW(28,24,9)
      dr_nnw = NNW.NNW(28,38,24)

      sp_nnw.setTrainData(NN_inp, NN_sp_oup)
      dr_nnw.setTrainData(NN_inp, NN_dr_oup)

      sp_nnw.trainData()
      dr_nnw.trainData()

      speed_para = sp_nnw.parameter()
      dir_para = dr_nnw.parameter()


    
    generation += 1      


    
  
