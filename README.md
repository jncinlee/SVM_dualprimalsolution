# SVM_dual primal solution
Provide dual solution for linear SVM and compare its run time performance on MNIST dataset from http://yann.lecun.com/exdb/mnist/. We first state the primal problem of hard margin linear SVM as

$$\min_{w,b} ||w||^2 $$
subject to $$y_i(w^Tx_i+b) \leq 1,$$ for $$i = 1,...,n$$

By Lagrangian function, coefficient $$\alpha$$ and fulfilling KKT(Karush-Kuhn-Tucker) conditions, we could solve for dual problem that optimizing $$\alpha$$ while minizing $$w, b$$ in the primal problem. That is written

$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{ij} \alpha_i \alpha_j y_i y_j x_i^T x_j$$
subject to $$\alpha_i \geq 0$$ and $$\sum_i \alpha_i y_i = 0$$ for $$i = 1,...,n$$

