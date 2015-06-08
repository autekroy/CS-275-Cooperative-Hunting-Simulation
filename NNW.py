import numpy
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer

# http://pybrain.org/docs/quickstart/dataset.html?highlight=neural%20network
from pybrain.datasets import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet

class NNW:
  def __init__(self, num_input, num_hidden, num_output):
      # self.net = buildNetwork(num_input, num_hidden, num_output, bias = True)
    self.net = FeedForwardNetwork()
    inLayer = LinearLayer(num_input, name='in')
    hiddenLayer1 = SigmoidLayer(num_hidden, name='hidden1')
    outLayer = LinearLayer(num_output, name='out')

    self.net.addInputModule(inLayer)
    self.net.addModule(hiddenLayer1)
    self.net.addOutputModule(outLayer)

    self.in_to_hidden = FullConnection(inLayer, hiddenLayer1)
    self.hidden_to_out = FullConnection(hiddenLayer1, outLayer)

    self.net.addConnection(self.in_to_hidden)
    self.net.addConnection(self.hidden_to_out)

    self.net.sortModules()

      # len(self.net.params)
      # len(in_to_hidden.params)
      # len(hidden_to_out.params)


  def activate(self, inputData):
      # self.net.sortModules()
      decision = self.net.activate(inputData)
      return decision

  def parameter(self, laynumber):
    if laynumber == 0:
          return slef.net.params
    elif laynumber == 1:
          return self.in_to_hidden.params
    elif laynumber == 2:
          return self.hidden_to_out.params

