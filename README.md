# SVM_dual primal solution
Provide dual solution for linear SVM and compare its run time performance on MNIST dataset from http://yann.lecun.com/exdb/mnist/. We first state the primal problem of hard margin linear SVM as

$$\min_{w,b} ||w||^2 $$
subject to $$y_i(w^Tx_i+b) \geq 1,$$ for $$i = 1,...,n$$

By Lagrangian function, coefficient $$\alpha$$ and fulfilling KKT(Karush-Kuhn-Tucker) conditions, we could solve for dual problem that optimizing $$\alpha$$ while minizing $$w, b$$ in the primal problem. That is written

$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{ij} \alpha_i \alpha_j y_i y_j x_i^T x_j$$
subject to $$\alpha_i \geq 0$$ and $$\sum_i \alpha_i y_i = 0$$ for $$i = 1,...,n$$

First import neccesary lib, MNIST dataset as training part and testing part with its label

```py
# Load libraries
import utils1,numpy,cvxopt,cvxopt.solvers,scipy,scipy.spatial,random,time
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

R = numpy.random.mtrand.RandomState(1234).permutation(len(z))[:1000]

Xtrain = Xtrain[R]
Ttrain = Ttrain[R]
Xtest
Ttest
```
Second by solving convex optimization problem by CVXOPT in primal problem

Thirdly by solving dual problem


