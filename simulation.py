#!/usr/bin/python
import animats
import sys  # sys.exit()
import pygame
import math

class Simulation:
  def __init__(self, num_preds, num_preys, width, height, saved_nets):
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
    self.pred_sprite   = pygame.transform.scale(self.pred_sprite, (20,20))
    self.bg            = pygame.transform.scale(self.bg, (1000, 700))
    self.prey          = pygame.transform.scale(self.prey, (20, 20))

    self.env = animats.Environment(num_preds, num_preys, width, height, saved_nets)

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
      self.screen.blit(self.prey, (prey.loc[0] - 10, prey.loc[1] - 10))

    # paint predator
    for pred in self.env.predators:
      self.screen.blit(pygame.transform.rotate(self.pred_sprite, 360 - pred.direction), (pred.loc[0] - animats.Predator.radius, pred.loc[1] - animats.Predator.radius))

    pygame.display.flip()

if __name__ == "__main__":
  # load save state from file
  filename = ""
  if len(sys.argv) > 1:
    filename = sys.argv[1]
  simulation = Simulation(3, 2, 1000, 700, filename)
  
  # main loop
  while 1: 
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