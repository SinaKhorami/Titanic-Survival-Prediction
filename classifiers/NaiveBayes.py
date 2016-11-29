#author: Sina Khorami
#date: Tue 29 Nov 2016
#NaiveBayes Classifier

import os
import sys
import csv

sys.path.append(os.path.abspath("../"))

from sklearn.naive_bayes import MultinomialNB
from data_model import Data

class NaiveBayes():
	"""docstring for NaiveBayes"""
	def __init__(self):
		self.result = None

	def classify(self):
		data = Data()
		train_data, test_data = data.getCleanData()
		nb = MultinomialNB()
		nb.fit(train_data[0::, 1::], train_data[0::, 0])
		self.result = nb.predict(test_data)
		self.saveResult()

	def saveResult(self):
		result_file = open("result/naivebayesmodel.csv", "wb")
		result_file_object = csv.writer(result_file)
		result_file_object.writerow(["PassengerId", "Survived"])
		passengerId = 892
		for survived in self.result:
			result_file_object.writerow([str(passengerId), str(int(survived))])
			passengerId += 1
		result_file.close()

		print "NaiveBayes Prediction Saved Successfully!"