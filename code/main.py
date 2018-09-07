import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.utils import shuffle
import os.path
import helper as hp
import featureextractor as fe
from sklearn import svm 
from sklearn.externals import joblib
import argparse

algo="all"
classes=3
writetofile=False
imbalance=False
parser = argparse.ArgumentParser(description='Choose algo and classes to run for, you can optionally choose to write output to csv')
parser.add_argument("algo",metavar='algo',help='Select algorithm, valid options are, unigram, bigram, tree, treemod or all')
parser.add_argument("classes",metavar='classes',help='Select number of classes, valid options are 2, 3')
parser.add_argument('--write', action='store_const', dest='writebool',const=True,help='use this option to write output to file') 
parser.add_argument('--imbalance', action='store_const', dest='imbalance',const=True,help='use this option to enable imbalanced data set')
args = parser.parse_args()
algo=args.algo
classes=int(args.classes)
filename="full-corpus"

if(args.writebool):
	writetofile=args.writebool
if(args.imbalance):
	filename = filename+"unmod"
	if(algo!="bigram" and algo!="unigram"):
		print("Imbalanced data set can only have unigram or bigram algo and 3 classes")
		exit()
	if(classes==2):
		print("Imbalanced data set can only have 3 classes")
		exit()
filename = filename+".csv"

if(algo!="unigram" and algo!="bigram" and algo!="tree" and algo!="treemod"):
	algo="all"


if(classes!=3 and classes!=2):
	classes=3

mapper = {"positive":1,"negative":0,"neutral":2}
if(classes==2):
	classname="Binary"
else:
	classname="Tertiary"
def data(classes):
	file1 = "../data/"+filename
	data = pd.read_csv(file1)
	data = data[data["Sentiment"]!="irrelevant"]
	if(classes==2):
		data = data[data["Sentiment"]!="neutral"]
	data["Sentiment"]=(data["Sentiment"].map(mapper))
	X=data["TweetText"].as_matrix()
	Y=data["Sentiment"].as_matrix()
	return X,Y
def create_ngram_features(X,num):
	
	vectorizer = CountVectorizer(min_df=1,ngram_range=(num,num))
	X = vectorizer.fit_transform(X)
	X = X.toarray()
	return X

#############################################################
#ngrams

if(algo=="unigram" or algo=="all"):
	num=1
	text,y = data(classes)
	X = create_ngram_features(text,num)
	hp.classify("svm.LinearSVC()",X,y,"Unigram "+classname,classes,True,writetofile,True)
if(algo=="bigram" or algo=="all"):
        num=2
        text,y = data(classes)
        X = create_ngram_features(text,num)
        hp.classify("svm.LinearSVC()",X,y,"Bigram "+classname,classes,True,writetofile,True)
if(algo=="tree" or algo=="all"):
	text,y = data(classes)
	param=""
	X=[]
	for index,treed in enumerate(text):
        	X.append([index])
	if(classes==2):
		param = "bigramorig"
		hp.classify("mysvm('"+param+"',60)",X,y,"Tree Kernel "+classname,classes,True,writetofile,True)
	else:
		param = "bigramorigc3"
		hp.classify("mysvm('"+param+"',60)",X,y,"Tree Kernel "+classname,classes,True,writetofile,True)
if(algo=="treemod" or algo=="all"):
        text,y = data(classes)
        param=""
        X=[]
        for index,treed in enumerate(text):
                X.append([index])
        if(classes==2):
                param = "bigrammod"
        	hp.classify("mysvm('"+param+"',30)",X,y,"Tree Kernel Mod "+classname,classes,True,writetofile,True)
	else:
                param = "bigrammodc3"
        	hp.classify("mysvm('"+param+"',10)",X,y,"Tree Kernel Mod "+classname,classes,True,writetofile,True)


#sentifeatures
#text,y = data(classes)
#sent="@Fernando this isn't a great day for playing the HARP! :) http://lucidroit.com! cooool brb"
#sents=[]
#sents.append(sent)
#print(sent)
#fe.makepseudotree(text)



