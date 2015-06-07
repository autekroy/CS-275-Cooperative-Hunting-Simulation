import numpy
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer


class NNW:
	def __init__(self, num_input, num_hidden, num_output):
      # self.net = buildNetwork(num_input, num_hidden, num_output, bias = True)
      self.net = FeedForwardNetwork()
      
      self.net.addInputModule(LinearLayer(num_input, name='in'))
      self.net.addModule(SigmoidLayer(num_hidden, name='hidden'))
      self.net.addOutputModule(LinearLayer(num_output, name='out'))

      in_to_hidden = FullConnection(inLayer, hiddenLayer)
      hidden_to_out = FullConnection(hiddenLayer, outLayer)
      
      self.net.addConnection(FullConnection(self.net['in'], self.net['hidden']))
      self.net.addConnection(FullConnection(self.net['hidden'], self.net['out']))
      self.net.sortModules()

    def activate(self, inputData):
      # self.net.sortModules()
      decision = self.net.activate(inputDaa)
      return decision