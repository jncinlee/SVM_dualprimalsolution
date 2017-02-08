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