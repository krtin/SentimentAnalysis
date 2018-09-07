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
algo="ngram"

mapper = {"positive":1,"negative":0,"neutral":2}

classes=2

filename = "../data/trees_moded_2classes.txt"
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
        #print(np.sum(data["Sentiment"]=="negative"))
	#print(np.sum(data["Sentiment"]=="positive"))
	#print(np.sum(data["Sentiment"]=="neutral"))
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

def treekernel(X,Y):
        R = np.zeros((len(X), len(Y)))
        for x in X:
                for y in Y:
                        i = int(x[0])
                        j = int(y[0])
                        print(i,j)
                        R[i,j] = getscore(i,j)  
                        print(R[i,j])
        joblib.dump(R, 'bigram_score_mod.pkl')
        return R  

text,y = data(classes)

clf = svm.SVC(kernel=treekernel)
clf.fit(X, y)
joblib.dump(clf, 'bigram_svm_mod.pkl') 
