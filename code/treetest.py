
from sklearn import svm
import tree_kernels 
import tree 
#lambda 

# kernel parameter 
dat = tree.Dataset() 
dat.loadFromFilePrologFormat("../data/trees.txt") 
#print(dat.examples[0].getDepth())
k = tree_kernels.KernelPT(0.2,0.3) 
a=k.preProcess(dat.getExample(0))
b=k.preProcess(dat.getExample(1))
print(a)
print(b)
#print(a)
ans=k.evaluate(a,b)
#ans=k.evaluate(dat.getExample(0),dat.getExample(1))
print(ans)
clf = svm.SVC(kernel=my_kernel)
clf.fit(dat, Y)
#k.printKernelMatrix(dat) 
