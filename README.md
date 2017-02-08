# SVM_dual primal solution
Provide dual solution for linear SVM and compare its run time performance on MNIST dataset from http://yann.lecun.com/exdb/mnist/. We first state the primal problem of hard margin linear SVM as

$$\min_{w,b} ||w||^2 $$
subject to $$y_i(w^Tx_i+b) \geq 1,$$ for $$i = 1,...,n$$

By Lagrangian function, coefficient $$\alpha$$ and fulfilling KKT(Karush-Kuhn-Tucker) conditions, we could solve for dual problem that optimizing $$\alpha$$ while minizing $$w, b$$ in the primal problem. That is written

$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{ij} \alpha_i \alpha_j y_i y_j x_i^T x_j$$
subject to $$\alpha_i \geq 0$$ and $$\sum_i \alpha_i y_i = 0$$ for $$i = 1,...,n$$

## First import neccesary lib, MNIST dataset as training part and testing part with its label

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
## Second by solving convex optimization problem by CVXOPT in primal problem

our objective function and restraint is, vector label pairs denote as $(x_i, y_i)$ for label $y_i \in (1, -1)$ for $i = 1...N$
\begin{equation*}
\begin{aligned}
    & \min_{w,b} & ||w||^2  \\
    & \mbox{s.t } & y_i(w^Tx_i + b) \geq 1 \\
\end{aligned}
\end{equation*}

And could be rewritten as optimization problem of CVXOPT as below,
\begin{equation*}
\begin{aligned}
    & \min_{\alpha} & \frac{1}{2}\alpha^T P \alpha \\
    & \mbox{s.t } & -1 \times \left(\mathbf{X} | \mathbf{1} \right) \alpha \leq \frac{\mathbf{1}}{y} \\
\end{aligned}
\end{equation*}

where $\alpha 
= \left( \begin{array}{c}
w_1     \\
\vdots  \\
w_{748} \\
b  \end{array} \right)$ and $\mathbf{1}$ element-wise divided by vector $y$

```py
cvxopt.solvers.options['show_progress'] = False
case_n = len(Xtrain)   #1000
dim_n = len(Xtrain[0]) #784

# Prepare the matrices for the quadratic program
def getPrim(Z,y):
    case_n = len(Z)   #1000
    dim_n = len(Z[0]) #784
    P1 = numpy.asarray(numpy.diag(numpy.ones(dim_n)))
    P2 = numpy.zeros([1, dim_n]) #collect P2
    P3 = numpy.zeros([1, 1])
    Pup = numpy.concatenate((P1,P2.T),axis = 1)
    Pdown = numpy.concatenate((P2,P3),axis = 1)
    P = cvxopt.matrix(numpy.concatenate((Pup,Pdown), axis = 0))
    q = cvxopt.matrix(numpy.zeros([dim_n +1 ,1]))
    G = cvxopt.matrix(numpy.concatenate((Z*-1.,numpy.ones([case_n,1])*-1.), axis = 1))     #combine G1 G2
    h = cvxopt.matrix(numpy.zeros(case_n)/y)
    A = cvxopt.matrix(numpy.zeros([1, dim_n +1]))
    b = cvxopt.matrix(0.0)
    return P,q,G,h,A,b
        
P,q,G,h,A,b = getPrim(Xtrain, Ttrain)
                
# Train the model (i.e. compute the alphas)
alpha = numpy.array(cvxopt.solvers.qp(P,q,G,h)['x']).flatten()

w_prim = alpha[:dim_n]
b_prim = alpha[dim_n:dim_n+1]

print alpha.shape
print w_prim.shape, b_prim
```

## Thirdly by solving dual problem on CVXOPT

The objective for the corresponding dual problem is to minimize the objective function, and state as matrix format as below,
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2} \sum_{ij} \alpha_i \alpha_j y_i y_j x_i^T x_j$$
subject to $$\alpha_i \geq 0$$ and $$\sum_i \alpha_i y_i = 0 \mbox{ for } i = 1,...,n$$

\begin{equation*}
\begin{aligned}
    & \min_{0 \leq \beta_i \leq C} & \frac{1}{2} \alpha^T \mathbf{H} \alpha - \mathbf{1}^T \alpha \\
    & \mbox{s.t } & \alpha \geq \mathbf{0} \\
    & \mbox{where} & \mathbf{H} = yy^T \odot XX^T
\end{aligned}
\end{equation*}

Then, given the $\alpha$, the parameter of the SVM can be obtained as:
$$
w = \sum_i \alpha_i y_i x_i
$$
where $b$ can generate as
$$
b = \frac{1}{\# SV} \sum_{i \in SV} \left( y_i - \sum_{j=1}^n \alpha_j y_j x_i^T x_j \right) 
$$
and `SV` is the set of indices corresponding to the unbound support vectors.

```py
cvxopt.solvers.options['show_progress'] = False

# Prepare the matrices for the quadratic program
def getDual(Z,y):
    nb = len(Z)
    nt = len(y)
    P = cvxopt.matrix((numpy.outer(y,y) * numpy.dot(Z,Z.T)))
    #P = cvxopt.matrix(((numpy.outer(Ttrain,Ttrain))*(numpy.dot(Xtrain,Xtrain.T)*-1)))
    q = cvxopt.matrix(numpy.ones(nb)*-1.,(nb,1))
    G = cvxopt.matrix(numpy.diag(numpy.ones(nb)* -1.)) 
    h = cvxopt.matrix(numpy.zeros(nb))
    A = cvxopt.matrix(y,(1,nt))
    b = cvxopt.matrix(0.0)
    return P,q,G,h,A,b
        
P,q,G,h,A,b = getDual(Xtrain,Ttrain)
                
# Train the model (i.e. compute the alphas)
alpha = numpy.array(cvxopt.solvers.qp(P,q,G,h,A,b)['x']).flatten()

w_dual = numpy.dot(Xtrain.T,Ttrain*alpha)

SV = (alpha>1e-6)
uSV = SV*(alpha<1e-6)
b_dual = 1.0/(sum(uSV)+10^-10)*(Ttrain[uSV]-numpy.dot(numpy.dot(Xtrain[uSV,:],Xtrain.T),alpha*Ttrain)).sum()

print alpha.shape
print w_dual.shape, b_dual
```

## Result of run time comparison

If we change to soft-margin SVM with slack variable C, and changing the notation of label. We could generate a run time comparison betwenn Primal solution and Dual solution. The run time for Dual solution is almost one-half to the Primal, which makes it a more convenient way of solving SVM.

![alt tag](https://github.com/jncinlee/SVM_dualprimalsolution/blob/master/compare%20dual%20primal.png "Primal Dual run time comparison")
