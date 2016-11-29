import os
import sys

sys.path.append(os.path.abspath("classifiers/"))

from NaiveBayes import NaiveBayes
from RandomForest import RandomForest
from AdaBoost import AdaBoost

if __name__ == '__main__':
	nb = NaiveBayes() # 0.65072
	nb.classify()
	
	rf = RandomForest(100) # 0.72727
	rf.classify()

	ab = AdaBoost(50) # 0.76555
	ab.classify()