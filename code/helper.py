import pandas as pd
import numpy as np  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 
from sklearn.utils import shuffle
import os.path
from treemodel import mysvm 

def evaluate(yp,y,modelname,classes):
        CM=np.zeros((classes,classes)).astype(float)
        for m in range(0,len(y)):
                CM[y[m]][yp[m]]=CM[y[m]][yp[m]]+1
        acc = np.trace(CM)/len(y)
        recall=[]
        precision=[]
        f1=[]
        for i in range(0,classes):
                precision.append(float(CM[i][i])/float(np.sum(CM[:,i])))
                recall.append(float(CM[i][i])/float(np.sum(CM[i])))
                f1.append(2*precision[i]*recall[i]/(precision[i]+recall[i]))
        return [modelname, np.array_str(CM), acc, np.mean(precision), np.mean(recall), np.mean(f1)]

def crossvalidate(x_train,y_train,modelname,classes):
        k=5 
        CM=np.zeros((classes,classes)).astype(float)
        CM_train=np.zeros((classes,classes)).astype(float)
        datasets = np.array_split(x_train,k,axis=0)
        testsets = np.array_split(y_train,k,axis=0) 
        for i in range(0,k):
                learnon = np.concatenate((datasets[i % k], datasets[(i + 1) % k], datasets[(i + 2) % k], datasets[(i + 3) % k]),axis=0) 
                learny = np.concatenate((testsets[i % k], testsets[(i + 1) % k], testsets[(i + 2) % k], testsets[(i + 3) % k]),axis=0)
                teston = datasets[(i + 4) % k]
                testy = testsets[(i + 4) % k]
                clf = eval(modelname)
                clf.fit(learnon, learny)
                yp = clf.predict(teston)
                for m in range(0,len(testy)):
                        CM[testy[m]][yp[m]]=CM[testy[m]][yp[m]]+1    
        clf = eval(modelname)
        clf.fit(x_train, y_train)
        yt=clf.predict(x_train)
        for m in range(0,len(y_train)):
                CM_train[y_train[m]][yt[m]]=CM[y_train[m]][yt[m]]+1
        acc = np.trace(CM)/len(y_train)
        recall=[]
        precision=[]
        f1=[]
        for i in range(0,classes):
                precision.append(float(CM[i][i])/float(np.sum(CM[:,i])))
                recall.append(float(CM[i][i])/float(np.sum(CM[i])))
                f1.append(2*precision[i]*recall[i]/(precision[i]+recall[i]))
        acc_train = np.trace(CM_train)/len(y_train)
        #print("CV",acc)
        #print("Train",acc_train)
        return [modelname, np.array_str(CM), acc, np.mean(precision), np.mean(recall), np.mean(f1)]


def classify(modelname,X,y,specialname,classes,testbool=False,writebool=False,overwrite=False):
        clf = eval(modelname)
        X,y = shuffle(X,y,random_state=17)
        total=len(y)
        train=np.floor(total*0.8).astype("int")
        test=total-train
        X_train=X[:train]
        y_train=y[:train]
        X_test=X[train:]
        y_test=y[train:]
        clf.fit(X_train, y_train)
        data = []
        includeheader=False
	if(os.path.isfile("../data/results.csv") is False):
		includeheader = True
        header=["Model", "CM", "Accuracy", "Precision", "Recall", "F1","Name", "Type","Size"]
                #data.append(header)
        #cross validation error
        if(testbool is False or overwrite is True):
                row = crossvalidate(X_train,y_train,modelname,classes)
                row.append(specialname)
                row.append("CV")
                row.append(train)
                data.append(row)
        #test error
        if(testbool is True):
		y_pred = clf.predict(X_test) 
                row =  evaluate(y_pred,y_test,modelname,classes)
                row.append(specialname)
                row.append("Test")
                row.append(test)    
                data.append(row)
	data = pd.DataFrame(data)
	data.columns=header
	print(data)        
        if(writebool):
                #data = pd.DataFrame(data)
                with open('../results.csv', 'a') as f:
                        data.to_csv(f,header=includeheader,index=False)
