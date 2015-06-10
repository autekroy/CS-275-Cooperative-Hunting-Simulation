

# From --------------------------------------------------
# UCLA CS 260 - Machine Learning Algorithm
# Yao-Jen Chang, 2015 @UCLA
# Email: autekwing@ucla.edu
#
# Functions about loading training data
#--------------------------------------------------


# Read the trainning data
# Have 39 users with 20 label 0, 19 label 1
# Each user data have time series data from 26 week

# userData[0] has two list, which mean Diastolic and Systolic
# each list has 26 data point

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
