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

    self.num_input = num_input
    self.num_hidden = num_hidden
    self.num_output = num_output

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

    self.dataset = None


    # len(self.net.params)
    # len(in_to_hidden.params)
    # len(hidden_to_out.params)

  def train(self, learningRate = 0.1, batch = True, maxEpochs = 100, continueEpochs = 10):
    # http://pybrain.org/docs/api/supervised/trainers.html?highlight=backproptrainer#pybrain.supervised.trainers.BackpropTrainer
    # BackpropTrainer(module, dataset=None, learningrate=0.01, lrdecay=1.0, momentum=0.0, verbose=False, batchlearning=False, weightdecay=0.0)
    # things for setting:
    # 1. dataset
    # 2. learningrate: 0.01 ~ 0.25
    # 3. batchlearning: True or False
    trainer = BackpropTrainer(net, self.dataset, learningRate, batch)

    # trainUntilConvergence(dataset=None, maxEpochs=None, verbose=None, continueEpochs=10, validationProportion=0.25)
    # things for setting:
    # 1. maxEpochs: at most that many epochs are trained. 
    # 2. continueEpochs: Each time validation error hits a minimum, try for continueEpochs epochs to find a better one.
    # 3. validationProportion: ratio of the dataset for validation dataset.
    trainer.trainUntilConvergence()

  def setTrainData(train, target):
    ds = SupervisedDataSet(self.num_input, self.num_output)
    dataSize = len(train) # should be same as len(target)
    
    # example
    # ds.addSample((0, 0), (0,))
    # ds.addSample((0, 1), (1,))
    # ds.addSample((1, 0), (1,))
    # ds.addSample((1, 1), (0,))
    for i in range(dataSize):
      ds.addSample(train[i], target[i])

    self.dataset = ds


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

