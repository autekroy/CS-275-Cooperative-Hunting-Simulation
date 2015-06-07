import numpy
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer

# http://pybrain.org/docs/quickstart/dataset.html?highlight=neural%20network
from pybrain.datasets import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet
ds = ClassificationDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))
ds['input']
ds['target']


#net = buildNetwork(2, 3, 1, 2,4)
net = FeedForwardNetwork()


inLayer = LinearLayer(3, name='in')
hiddenLayer1 = SigmoidLayer(4, name='hidden1')
outLayer = LinearLayer(2, name='out')

net.addInputModule(inLayer)
net.addModule(hiddenLayer1)
net.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer1)
hidden_to_out = FullConnection(hiddenLayer1, outLayer)

net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)

net.sortModules()

net['in']
net['out']
net['hidden1']

len(net.params)
len(in_to_hidden.params)
len(hidden_to_out.params)

inputData = [1, 2, 1.5]
net.activate(inputData)
print net.params

# http://pybrain.org/docs/api/supervised/trainers.html?highlight=backproptrainer#pybrain.supervised.trainers.BackpropTrainer
# BackpropTrainer(module, dataset=None, learningrate=0.01, lrdecay=1.0, momentum=0.0, verbose=False, batchlearning=False, weightdecay=0.0)
# things for setting:
# 1. dataset
# 2. learningrate: 0.01 ~ 0.25
# 3. batchlearning: True or False

trainer = BackpropTrainer(net, ds)

# training
# train() Train the associated module for one epoch.
trainer.train()
print net.params

# trainUntilConvergence(dataset=None, maxEpochs=None, verbose=None, continueEpochs=10, validationProportion=0.25)
# things for setting:
# 1. maxEpochs: at most that many epochs are trained. 
# 2. continueEpochs: Each time validation error hits a minimum, try for continueEpochs epochs to find a better one.
# 3. validationProportion: ratio of the dataset for validation dataset.

trainer.trainUntilConvergence()
print net.params



# http://pybrain.org/docs/api/tools.html?highlight=regression#pybrain.tools.neuralnets.NNregression
# check NNregression
