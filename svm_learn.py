#!/usr/bin/python
import animats
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


def train(training):
	size = len(training[0])
	situations = []
	behaviors = []
	i = 0
	while i < len(training):
		situations.append(training[i][0:size-1])
		behaviors.append(training[i][size-1])
		i+=1
	X = np.array(situations)
	y = np.array(behaviors)
	clf = OneVsRestClassifier(SVC(kernel='poly', degree=3))
	clf.fit(X,y)
	return clf
		

def learn(lion1_pos, lion2_pos, zebra1_pos, zebra2_pos, self_behav, lion1_behav, lion2_behav, zebra1_behav, zebra2_behav):
	return train(selected_lions).predict(np.array([[lion1_pos, lion2_pos, zebra1_pos, zebra2_pos, self_behav, lion1_behav, lion2_behav, zebra1_behav, zebra2_behav]]))


	


