
# ============================== Test for simulation ============================

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
from simulation import *

(InputSamples, SpeedSamples, DirectionSamples, Fitness) = ReadSampleData("sampleData")

input, speed, dir = readTrainData(Fitness,2)

Init_Speed_Net = NNW.NNW(30,42,9)
Init_Dir_Net = NNW.NNW(30,42,24)
Init_Speed_Net.setTrainData(InputSamples,SpeedSamples)
Init_Dir_Net.setTrainData(InputSamples,DirectionSamples)

Init_Speed_Net.trainOnce()
Init_Dir_Net.trainOnce()

Init_Speed_Net.trainData()
Init_Dir_Net.trainData()

# ============================== Test for reading file ============================
from readFile import *
import NNW
from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

sampleTrain, sampleTarget1, sampleTarget2 = readData("sample/data")

sample_seed_net = NNW.NNW(28,24,9)
sample_dir_net = NNW.NNW(28,38,24)

sampleTrain, sampleTarget1, sampleTarget2 = readData("sample/data")

sample_seed_net = NNW.NNW(28,24,9)
sample_dir_net = NNW.NNW(28,38,24)

sample_seed_net.setTrainData(sampleTrain, sampleTarget1)
sample_dir_net.setTrainData(sampleTrain, sampleTarget2)

sample_seed_net.parameter()
sample_dir_net.parameter()


sample_seed_net.trainOnce()
sample_dir_net.trainOnce()

sample_seed_net.trainData()
sample_dir_net.trainData()


# ============================== Test for NNW ============================

import numpy
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import RecurrentNetwork, FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer

# http://pybrain.org/docs/quickstart/dataset.html?highlight=neural%20network
from pybrain.datasets import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))
ds['input']
ds['target']


#net = buildNetwork(2, 3, 1, 2,4)
net = FeedForwardNetwork()


inLayer = LinearLayer(2, name='in')
hiddenLayer1 = SigmoidLayer(3, name='hidden1')
outLayer = LinearLayer(1, name='out')

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

inputData = [1, 0]
net.activate(inputData)
net.activate([1,0])
print net.params

# http://pybrain.org/docs/api/supervised/trainers.html?highlight=backproptrainer#pybrain.supervised.trainers.BackpropTrainer
# BackpropTrainer(module, dataset=None, learningrate=0.01, lrdecay=1.0, momentum=0.0, verbose=False, batchlearning=False, weightdecay=0.0)
# things for setting:
# 1. dataset
# 2. learningrate: 0.01 ~ 0.25
# 3. batchlearning: True or False

# check the dimension is same
ds.indim == network.indim

trainer = BackpropTrainer(net, ds, learningrate = 0.1, batchlearning=True)

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
