#!/usr/bin/python
import animats
import sys  # sys.exit()
import pygame
import math
import os

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

    pygame.display.flip()



def get_last_line(file):
  f = open(file,'r')
  line1 = None
  line2 = f.readline()
  while line2:
    line1 = line2
    line2 = f.readline()
  return line1

fitness = []



if __name__ == "__main__":
  # load save state from file
  generation = 0
  iter_num = 0
  max_iter = 1
  filename = ""

  if len(sys.argv) > 2:
    filename = "training_data"
    generations = int(sys.argv[1])
    max_iter = int(sys.argv[2])

  while generation < generations:

    j = 0
    while j < len(fitness):
      fitness.pop()
      j += 1;

    simulation = Simulation(generation, 3, 1, 1000, 700, filename+'_gen_'+str(generation)+'_iter_'+str(iter_num)+'.csv')
  
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
        if len(fitness)<5: 
          fitness.append((iter_num,fit))
          fitness.sort(lambda x,y:cmp(x[1],y[1]))
        elif fitness[0][1] < fit:
          fitness.pop(0)
          fitness.append((iter_num,fit))
          fitness.sort(lambda x,y:cmp(x[1],y[1]))
        print fitness

    
        iter_num += 1
        if iter_num < max_iter:
          simulation = Simulation(generation, 3, 1, 1000, 700, filename+'_gen_'+str(generation)+'_iter_'+str(iter_num)+'.csv')

    generation += 1      


    
  
