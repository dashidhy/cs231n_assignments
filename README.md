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
At the first time I did this homework, it really took me a long time to figure out the way of vectorlizing the computation of the gradient of the weight matrix. However, when I reviewed this problem again after finishing the Softmax classifier, with the intuition of computation graph and back propagation I found it's easy to reach the final expressions in my code. Here are the steps:

First stage:
```    
X -----------                                               
             -                                              
              -                 <---- dS = ...              
               --> S = X.dot(W) ----------->                      
              -                                             
             -                                              
W -----------                                               
  <---- dW = X.T.dot(dS)      
```
Second stage:

```
   <-- dS = dS_d                                  
S ----------------------------------------------------------------------------
   -                                                                          -
    - <---- dS = dSy*mask_y                                                    -       
     -         = -np.array([np.sum(dS_d, 1)]).T*mask_y                          -                  
      -                                                                          -                  <----dS_d = ...
       -                                                                          --> S_d = S-Sy_bc ----------->
        -                  <---- dSy = dSy_bc.dot(ones_c.T)                      -
         --> Sy = S*mask_y -----------  = -dS_d.dot(ones_c)                     -             
        -                             -                                        -                            
       -                               -                                      -
      -                                 --> Sy_bc = Sy.dot(ones_c) -----------                            
     -                                 -                           <---- dSy_bc = -dS_d  
    -                                 -
   -               ones_c -----------                                     
  -                = 
mask_y             np.ones((num_classes, num_classes))
= 
(np.array([y]).T == np.arange(num_classes))
```
Final stage:
```
    <---- dS_d = dS_h = S_0
S_d -----------
               -
                -                <---- dS_h = dS_loss*S_0 = S_0
                 --> S_h = S_d+1 -----------
                -                           -
               -                             -                     <---- dS_loss = np.ones(S_loss.shape)             
  1 -----------                               --> S_loss = S_h*S_0 -----------> loss = np.sum(S_loss)-1
                                             -
                                            -
       S_0 = (S_h > 0)/num_train -----------
```
Thus, merge the computation graphs together and then simplify the expressions, we get:
<div align=center><img src="https://latex.codecogs.com/svg.latex?%5Cfn_cm%20%24%24%20%5Cbegin%7Baligned%7D%20dW%20%26%20%3D%20X.T.dot%28S%5C_0%29-X.T.dot%28np.array%28%5Bnp.sum%28S%5C_0%2C%201%29%5D%29.T*mask%5C_y%29%20%5C%5C%20%26%3D%20X.T.dot%28S%5C_0%29-np.dot%28X.T*np.sum%28S%5C_0%2C%201%29%2C%20mask%5C_y%29%20%5Cend%7Baligned%7D%20%24%24"/></div>
A possible original code is the one below:
```python
def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  S = X.dot(W)
  sy = S[list(range(num_train)), y]
  S = (S-np.array([sy]).T)+1
  S_0 = (S > 0)/num_train
  loss = np.sum(S*S_0)-1+reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dW = 2*reg*W
  mask_y = (np.array([y]).T == np.arange(num_classes))
  dW += np.dot(X.T, S_0)-np.dot(X.T*np.sum(S_0, 1), mask_y)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
```
