# Some tips of cs231n spring 2017 assignments
Here I take some notes about some of my ideas that I think may be useful in my future programming.
# Assignment 1
## Q1: K-Nearest Neighbor Classifier
In the KNN problem, the most tricky part is how to fully vectorlize the computation of l2 distance. The solution is that first, **decompose the final distance matrix,** and second, **vectorlize the computing process with matrix multiplication and broadcast sums.**

Consider each element of the final distance matrix: 
<div align=center><img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%24%24%20%5Cbegin%7Baligned%7D%20dist%5Bi%2C%20j%5D%20%26%20%3D%20%5Csum_k%20%28X%5Bi%2Ck%5D-self.X%5C_train%5Bj%2Ck%5D%29%5E2%20%5C%5C%20%5C%5C%20%26%20%3D%20%5Csum_k%28X%5E2%5Bi%2Ck%5D&plus;self.X%5C_train%5E2%5Bj%2Ck%5D-2*X%5Bi%2Ck%5D*self.X%5C_train%5Bj%2Ck%5D%29%20%5Cend%7Baligned%7D%20%24%24"/></div>

In this way, we can decompose each element of the dist matrix into three terms. Each of the terms can be vectorlized into a Numpy form:
<div align=center><img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%24%24%20%5Cbegin%7Baligned%7D%20%5Csum_k%20X%5E2%5Bi%2Ck%5D%20%26%5Cto%20np.array%28%5Bnp.sum%28np.square%28X%29%2C%201%29%5D%29.T%20%5C%5C%20%5C%5C%20%5Csum_k%20X%5C_train%5E2%5Bj%2Ck%5D%20%26%5Cto%20np.sum%28np.square%28self.X%5C_train%29%2C%201%29%20%5C%5C%20%5C%5C%20%5Csum_k%20X%5Bi%2Ck%5D*self.X%5C_train%5Bj%2Ck%5D%20%26%5Cto%20X%5Bi%2Ck%5D*self.X%5C_train.T%5Bk%2Cj%5D%20%5Cto%20X.dot%28self.X%5C_train.T%29%20%5Cend%7Baligned%7D%20%24%24"/></div>

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
