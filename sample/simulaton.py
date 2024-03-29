#!/usr/bin/python
import animats
import sys  # sys.exit()
import pygame
import math
from time import sleep

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
    if (self.env.end_iteration()):
      sleep(1.5)
    return self.env.end_iteration()

  def texts(self, score, line, font_size):
    font=pygame.font.Font(None,font_size)
    text=font.render(str(score), 1,(0,0,0))
    self.screen.blit(text, (30, 30+int(line)*15)) 

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
      self.texts("PREDATOR " + str(count+1), count*5, 20)
      self.texts("  state: " + pred.direction_text, count*5 + 1, 20)
      self.texts("  direction: " + pred.speed_text, count*5 + 2, 20)
      self.texts("  energy: " + str(pred.energy), count*5 + 3, 20)
      count += 1
    
    pygame.display.flip()

if __name__ == "__main__":
  # load save state from file
  generation = 1
  iter_num = 0
  max_iter = 1
  filename = ""
  if len(sys.argv) > 2:
    filename = sys.argv[1]
    max_iter = int(sys.argv[2])
  simulation = Simulation(generation, 3, 1, 1000, 700, filename+str(iter_num)+'.csv')
  
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
      iter_num += 1
      if iter_num < max_iter:
        simulation = Simulation(generation, 3, 1, 1000, 700, filename+str(iter_num)+'.csv')
