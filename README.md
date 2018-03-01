# Some tips of cs231n spring 2017 assignments
Here I take some notes about some of my ideas that I think may be useful in my future programming.
# Assignment 1
## Q1: K-Nearest Neighbor Classifier
In the KNN problem, the most tricky part is how to fully vectorlize the computation of l2 distance. The solution is that first,** decompose the final distance matrix**, and second, **vectorlize the computing process with matrix multiplication and broadcast sums.**

Consider each element of the final distance matrix: 
<center><img src="https://latex.codecogs.com/png.latex?$$&space;\begin{aligned}&space;dist[i,&space;j]&space;&&space;=&space;\sum_k&space;(X[i,k]-self.X\_train[j,k])^2&space;\\&space;\\&space;&&space;=&space;\sum_k&space;X^2[i,k]&plus;self.X\_train^2[j,k]-2*X[i,k]*self.X\_train[j,k]&space;\end{aligned}&space;$$" title="$$ \begin{aligned} dist[i, j] & = \sum_k (X[i,k]-self.X\_train[j,k])^2 \\ \\ & = \sum_k X^2[i,k]+self.X\_train^2[j,k]-2*X[i,k]*self.X\_train[j,k] \end{aligned} $$" /></center>

In this way we decompose each element of the dist matrix into three terms. Each of the terms can be vectorlized into a Numpy form:
<div align: center>
<a href="https://www.codecogs.com/eqnedit.php?latex=$$&space;\begin{aligned}&space;\sum_k&space;X^2[i,k]&space;&\to&space;np.array([np.sum(np.square(X),&space;1)]).T&space;\\&space;\\&space;\sum_k&space;X\_train^2[j,k]&space;&\to&space;np.sum(np.square(self.X\_train),&space;1)&space;\\&space;\\&space;\sum_k&space;X[i,k]*self.X\_train[j,k]&space;&\to&space;X[i,k]*self.X\_train^T[k,j]&space;\to&space;X.dot(self.X\_train.T)&space;\end{aligned}&space;$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$&space;\begin{aligned}&space;\sum_k&space;X^2[i,k]&space;&\to&space;np.array([np.sum(np.square(X),&space;1)]).T&space;\\&space;\\&space;\sum_k&space;X\_train^2[j,k]&space;&\to&space;np.sum(np.square(self.X\_train),&space;1)&space;\\&space;\\&space;\sum_k&space;X[i,k]*self.X\_train[j,k]&space;&\to&space;X[i,k]*self.X\_train^T[k,j]&space;\to&space;X.dot(self.X\_train.T)&space;\end{aligned}&space;$$" title="$$ \begin{aligned} \sum_k X^2[i,k] &\to np.array([np.sum(np.square(X), 1)]).T \\ \\ \sum_k X\_train^2[j,k] &\to np.sum(np.square(self.X\_train), 1) \\ \\ \sum_k X[i,k]*self.X\_train[j,k] &\to X[i,k]*self.X\_train^T[k,j] \to X.dot(self.X\_train.T) \end{aligned} $$" /></a></div>

Thus, the fully vectorlized L2 distance code is something like the code below:

```Python
def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """ 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    X_temp = np.array([np.sum(np.square(X), 1)]).T
    X_trian_temp = np.sum(np.square(self.X_train), 1)
    dists = (-2*X.dot(self.X_train.T)+X_temp)+X_trian_temp
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists
```


## Q2: Training a Support Vector Machine
It really took me a long time to figure out the way of vectorlizing the computation of the gradient of the weight matrix. Here is my strategy:

Waiting to be completed...
