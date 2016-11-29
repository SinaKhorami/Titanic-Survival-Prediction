#author: Sina Khorami
#date: Tue 29 Nov 2016

import numpy as np
import pandas as pd

class Data():
	"""docstring for Data"""
	def __init__(self):
		self.train_df = pd.read_csv('../data/train.csv', header = 0)
		self.test_df = pd.read_csv('../data/test.csv', header = 0)

	def getCleanData(self):
		self.featureCreation()
		self.featureImprovement()
		self.featureSelection()
		return self.train_df.values, self.test_df.values

	def featureCreation(self):
		self.train_df['Gender'] = self.train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
		self.train_df['FamilySize'] = self.train_df['SibSp'] + self.train_df['Parch']

		self.test_df['Gender'] = self.test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
		self.test_df['FamilySize'] = self.test_df['SibSp'] + self.test_df['Parch']

	def featureImprovement(self):
		train_median_ages, test_median_ages = self.getMedianAgesByGenderAndPclass()
		self.train_df['AgeNotNan'] = self.train_df['Age']
		self.test_df['AgeNotNan'] = self.test_df['Age']
		for i in range(0, 2):
			for j in range(0, 3):
				self.train_df.loc[(self.train_df['Age'].isnull()) & (self.train_df['Gender'] == i) & \
					   			  (self.train_df['Pclass'] == j+1), 'AgeNotNan'] = train_median_ages[i, j]
				self.test_df.loc[(self.test_df['Age'].isnull()) & (self.test_df['Gender'] == i) & \
					   			  (self.test_df['Pclass'] == j+1), 'AgeNotNan'] = test_median_ages[i, j]
		self.train_df['Age'] = self.train_df['AgeNotNan']
		self.test_df['Age']  = self.test_df['AgeNotNan']

		self.train_df.loc[(self.train_df['Fare'].isnull()), 'Fare'] = self.train_df['Fare'].median()
		self.test_df.loc[(self.test_df['Fare'].isnull()), 'Fare'] = self.test_df['Fare'].median()

	def featureSelection(self):
		drop_elements = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin',\
				  		 'AgeNotNan', 'Embarked', 'SibSp', 'Parch']
		self.train_df = self.train_df.drop(drop_elements, axis = 1)
		self.test_df  = self.test_df.drop(drop_elements, axis = 1)

	def getMedianAgesByGenderAndPclass(self):
		train_median_ages = np.zeros((2, 3))
		test_median_ages = np.zeros((2, 3))
		for i in range(0, 2):
			for j in range(0, 3):
				train_median_ages[i, j] = self.train_df[(self.train_df['Gender'] == i) & \
									   	 (self.train_df['Pclass'] == j+1)]['Age'].dropna().median()
				test_median_ages[i, j]  = self.test_df[(self.test_df['Gender'] == i) & \
									   	 (self.test_df['Pclass'] == j+1)]['Age'].dropna().median()
		return train_median_ages, test_median_ages
