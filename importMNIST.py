# Load libraries
import numpy,cvxopt,cvxopt.solvers,scipy,scipy.spatial,random,time
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')    #get MNIST data set

Xtrain = mnist.data[:60000]
Xtest = mnist.data[60000:70000]
Ttrain = mnist.target[:60000]
Ttest = mnist.target[60000:70000]
    
    
Xtrain = Xtrain[(Ttrain==5)|(Ttrain==6)]
Ttrain = Ttrain[(Ttrain==5)|(Ttrain==6)]
Ttrain = 1.0*(Ttrain==5)-1.0*(Ttrain==6)   #label digit5 as 1, digit6 as -1

Xtest = Xtest[(Ttest==5)|(Ttest==6)]
Ttest = Ttest[(Ttest==5)|(Ttest==6)]
Ttest = 1.0*(Ttest==5)-1.0*(Ttest==6)

m = Xtrain.mean(axis=0)
Xtrain = Xtrain - m

s = Xtrain.std()
Xtrain = Xtrain / s

R = numpy.random.mtrand.RandomState(1234).permutation(len(Xtrain))[:1000]

Xtrain = Xtrain[R]
Ttrain = Ttrain[R]
Xtest
Ttest