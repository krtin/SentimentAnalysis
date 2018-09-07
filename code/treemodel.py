import pandas as pd
import numpy as np  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 
from sklearn.utils import shuffle
import os.path
import helper as hp
import featureextractor as fe
from sklearn import svm 
import tree_kernels 
import tree 
from sklearn.externals import joblib
import helper as hp
#mapper = {"positive":1,"negative":0,"neutral":2}
#classes=2
class mysvm():
	def __init__(self,algo,reg):
		filename=''
		
		if(algo=="bigramorig"):
			filename = 'bigram_score_orig.pkl'
		elif(algo=="bigrammod"):
			filename = 'bigram_score_mod.pkl'
		elif(algo=="bigramorigc3"):
			filename = 'bigram_score_c3.pkl'
		else:
			filename = 'bigram_score_mod_c3.pkl'				
		self.score = joblib.load(filename)
		self.reg=reg

	def fit(self,X,Y):	
		self.X=X
		self.Y=Y
		clf = svm.SVC(kernel=self.treekernel,shrinking=True,C=self.reg)
		clf.fit(X,Y)
		self.clf=clf

	def predict(self,X):
		return self.clf.predict(X)

	def treekernel(self,X,Y):
		score = self.score
		R = np.zeros([len(X), len(Y)])
        	for indexi,x in enumerate(X):
                	for indexj,y in enumerate(Y):
                        	i = int(x[0])
                        	j = int(y[0])
				R[indexi][indexj] = score[i][j]
		return R
