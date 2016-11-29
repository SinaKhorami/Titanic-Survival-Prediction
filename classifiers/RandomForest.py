#author: Sina Khorami
#date: Tue 29 Nov 2016
#RandomForest Classifier

import os
import sys
import csv

sys.path.append(os.path.abspath("../"))

from sklearn.ensemble import RandomForestClassifier
from data_model import Data

class RandomForest():
	"""docstring for RandomForest"""
	def __init__(self, num_estimators):
		self.n_estimators = num_estimators
		self.result = None

	def classify(self):
		data = Data()
		train_data, test_data = data.getCleanData()
		forest = RandomForestClassifier(n_estimators = self.n_estimators)
		forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
		self.result = forest.predict(test_data)
		self.saveResult()

	def saveResult(self):
		result_file = open("../result/randomforestmodel.csv", "wb")
		result_file_object = csv.writer(result_file)
		result_file_object.writerow(["PassengerId", "Survived"])
		passengerId = 892
		for survived in self.result:
			result_file_object.writerow([str(passengerId), str(int(survived))])
			passengerId += 1
		result_file.close()

		print "RandomForest Prediction Saved Successfully!"
