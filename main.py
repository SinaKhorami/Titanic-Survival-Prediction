#author: Sina Khorami
#date: Mon 28 Nov 2016

import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
	result = getResult()
	prediction_file = open("result/randomforestmodel_v1.csv", "wb")
	prediction_file_object = csv.writer(prediction_file)
	prediction_file_object.writerow(["PassengerId", "Survived"])
	passengerId = 892
	for survived in result:
		prediction_file_object.writerow([str(passengerId), str(int(survived))])
		passengerId += 1
	prediction_file.close()
	print "done!"

def getResult():
	train_data = getCleanData('train').values
	test_data = getCleanData('test').values
	forest = RandomForestClassifier(n_estimators = 80)
	forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
	output = forest.predict(test_data)
	return output

def getCleanData(data):
	if data == 'train':
		df = pd.read_csv('data/train.csv', header = 0)
	elif data == 'test':
		df = pd.read_csv('data/test.csv', header = 0)

	df = featureCreation(df)
	df = featureImprovement(df)
	df = featureSelection(df)
	return df

def featureSelection(df):
	df = df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin',\
				  'AgeNotNan', 'Embarked', 'SibSp', 'Parch'], axis = 1)
	return df

def featureImprovement(df):
	median_ages = getMedianAgesByGenderAndPclass(df)
	df['AgeNotNan'] = df['Age']
	for i in range(0, 2):
		for j in range(0, 3):
			df.loc[(df['Age'].isnull()) & (df['Gender'] == i) & \
				   (df['Pclass'] == j+1), 'AgeNotNan'] = median_ages[i, j]
	df['Age'] = df['AgeNotNan']

	df.loc[(df['Fare'].isnull()), 'Fare'] = df['Fare'].median()
	return df

def featureCreation(df):
	df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
	df['FamilySize'] = df['SibSp'] + df['Parch']
	return df

def getMedianAgesByGenderAndPclass(df):
	median_ages = np.zeros((2, 3))
	for i in range(0, 2):
		for j in range(0, 3):
			median_ages[i, j] = df[(df['Gender'] == i) & \
								   (df['Pclass'] == j+1)]['Age'].dropna().median()
	return median_ages

if __name__ == '__main__':
	main()