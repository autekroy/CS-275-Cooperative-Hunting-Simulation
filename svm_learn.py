#!/usr/bin/python
import animats
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


''' self is lion 0
 lion 1 position - self position
 lion 2 position - self position
 zibra 1 position - self position
 zibra 2 position - self postion
 self Behavior
 lion 1 Behavior
 lion 2 Behavior
 zibra 1 behavior
 zibra 2 behavior
'''

def train(inp, outp):
	X = np.array(inp)
	y = np.array(outp)
	clf = OneVsRestClassifier(SVC(kernel='poly', degree=3))
	clf.fit(X,y)
	return clf
		

def learn(lion1_posx, lion1_posy, lion2_posx, lion2_posy, zebra1_posx, zebra1_posy, zebra2_posx, zebra2_posy, lion1_behav, lion2_behav, zebra1_behav, zebra2_behav):
	return train(selected_lions).predict(np.array([[lion1_posx, lion1_posy, lion2_posx, lion2_posy, zebra1_posx, zebra1_posy, zebra2_posx, zebra2_posy, lion1_behav, lion2_behav, zebra1_behav, zebra2_behav]]))


	


