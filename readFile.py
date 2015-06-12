
import math
import os

def ReadSampleData(dir_path):
  InputSamples = []
  SpeedSamples = []
  DirectionSamples = []
  Fitness = []
  files = os.listdir(dir_path)
  for sample_file in files:
    if sample_file[-4:] != '.csv':
      continue
    file_path = dir_path + '/' + sample_file
    f = open(file_path,'r')
    if f.closed:
      continue
    for row in f:
      row = row.replace("\n", "")
      data = row.split(',')
      InputSamples.append(tuple(data[0:30]))
      SpeedSamples.append(tuple(data[30:39]))
      DirectionSamples.append(tuple(data[39:63]))
    f.close()
    Fitness.append((data[-1],file_path))
  return (InputSamples, SpeedSamples, DirectionSamples, Fitness)


def readData(path):
	import os
	train = []
	target1 = []
	target2 = []

	files = os.listdir('./' + path)
	for filename in files:
		print "read " + filename
		if filename[-4:] != '.csv': # if file is not csv type, jump
			continue

		inputSample, target1Sample, target2Sample = readcsvData(path + '/' + filename)
		
		for x in inputSample:
			train.append(x)

		for x in target1Sample:
			target1.append(x)

		for x in target2Sample:
			target2.append(x)

	return train, target1, target2


def readcsvData(filePath):
	import csv
	
	f = open(filePath, 'r')

	inputSample = []
	target1Sample = []
	target2Sample = []

	row = f.readline()

	while row:
		row = row.replace("\n", "")
		data = row.split(',')
		for i in range(3):
			data[i*9+0] = float(data[i*9+0])/10
			data[i*9+1] = float(data[i*9+0])/1000
			v_len = math.sqrt(float(data[i*9+3])**2 + float(data[i*9+4])**2)
			data[i*9+3] = float(data[i*9+3]) / v_len
			data[i*9+4] = float(data[i*9+4]) / v_len
			v_len = math.sqrt(float(data[i*9+5])**2 + float(data[i*9+6])**2)
			data[i*9+5] = float(data[i*9+5]) / v_len
			data[i*9+6] = float(data[i*9+6]) / v_len
			v_len = math.sqrt(float(data[i*9+7])**2 + float(data[i*9+8])**2)
			data[i*9+7] = float(data[i*9+7]) / v_len
			data[i*9+8] = float(data[i*9+8]) / v_len
		inputSample.append(tuple(data[0:28]))
		target1Sample.append(tuple(data[28:37]))
		target2Sample.append(tuple(data[37:61]))

		row = f.readline()

	return inputSample, target1Sample, target2Sample 


def listUserdate(list):
	# list will have 2 list with same size
	size = len(list[0]) # same as len(list[1])
	for i in range(size):
		print str(list[0][i]) + ',  ' + str(list[1][i])
