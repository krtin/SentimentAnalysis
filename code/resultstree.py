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

class mysvm():
	def __init__(self,algo):
		filename=''
		if(algo=="bigramorig"):
			filename = 'bigram_score_orig.pkl'
		
		self.score = joblib.load(filename)

	def fit(self,X,Y,cutlen=0):
		if(cutlen==0):
			score_train = self.score[:len(X),:len(X)]		
		#print(score_train.shape)
		self.createmodel(score_train,X,Y)	
	
	def createmodel(self,score,X,Y):
		self.currentscore=score
		clf = svm.SVC(kernel=self.treekernel)
		clf.fit(X,Y)
		self.clf=clf

	def predict(self,X,cutlen=0):
		if(cutlen==0):
			lent = len(self.score)-len(X)
			score_test = self.score[lent:,lent:]
			#print(score_test)
			#print(len(score_test))	
		yp=self.getprediction(X,score_test)
		return yp

	def getprediction(self,X,score):
		#self.currentscore=score
		clf=self.clf
		return clf.predict(X)

	def treekernel(self,X,Y):
		return self.currentscore

	
algo="ngram"

mapper = {"positive":1,"negative":0,"neutral":2}

classes=2

filename = "../data/trees_"+str(classes)+"classes.txt"
X=[]

with open(filename) as f:
    treedata = f.read().splitlines() 

for index,treed in enumerate(treedata):
        X.append([index])

def data(classes):
        file1 = "../data/sanders.csv"
        data = pd.read_csv(file1)
        data = data[data["Sentiment"]!="irrelevant"]
        if(classes==2):
                data = data[data["Sentiment"]!="neutral"]
        data["Sentiment"]=(data["Sentiment"].map(mapper))
        X=data["TweetText"].as_matrix()
        Y=data["Sentiment"].as_matrix()
        return X,Y 
 
def getscore(i,j):
        k = tree_kernels.KernelPT(0.2,0.3)
        dat = tree.Dataset() 
    
        a=dat.loadExamplePrologFormat(treedata[i])
        #print(a)
        b=dat.loadExamplePrologFormat(treedata[j])
        a=k.preProcess(a)
        b=k.preProcess(b)
        ans=k.evaluate(a,b)
        return ans 

score = joblib.load('bigram_score_orig.pkl')
def treekernel(X,Y):
	return score

text,y = data(classes)

totallen=len(X)
trainlen=int(len(X)*0.8)

X_test = X[:trainlen][:]
#print(X_test)
y_test = y[:trainlen]
#print(y_test)

clf=mysvm("bigramorig")
clf.fit(X_test,y_test)

y_p=clf.predict(X_test)

#clf=joblib.load('bigram_svm_orig.pkl')

#y_p=clf.predict(X)

trainresults = hp.evaluate(y_p,y_test,"Tree Kernel",classes)
print(trainresults)
#print(len(y))
#print(len(y_p))

#print(y)
#print(y_p)
