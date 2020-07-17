# Some tips of cs231n spring 2017 assignments

Here I take some notes of ideas that are tricky and useful.

---

<br/>

# Assignment 1

## Q1: k-Nearest Neighbor Classifier

In the KNN problem, the most tricky part is how to fully vectorlize the computation of l2 distance. The solution is that first, decompose the final distance matrix, and second, vectorlize the computing process with matrix multiplication and broadcast sums.

Consider each element of the final distance matrix: 

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f1.svg?sanitize=true"/></div>

<br/>In this way, we can decompose each element of the dist matrix into three terms. Each of the terms can be vectorlized into a Numpy form:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f2.svg?sanitize=true"/></div>

<br/>Thus, the fully vectorlized L2 distance code:

```Python
class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        self.X_train_square = np.sum(np.square(self.X_train), axis=1) # pre-calculate
    
    def compute_distances_no_loops(self, X):
        return np.sum(np.square(X), axis=1, keepdims=True) - 2.0 * X.dot(self.X_train.T) + self.X_train_square
```

<br/>

## Q2: Training a Support Vector Machine

At the first time I did this homework, it really took me a long time to figure out how to vectorlize the gradient computation. Thanks to  <a href="https://zhuanlan.zhihu.com/p/24709748" target="_blank">this article </a> from Zhihu, the computation becomes much easier by using matrix calculus methods. 

Total derivative of a matrix variable:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f3.svg?sanitize=true"/></div>

<br/>Trace tricks:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f4.svg?sanitize=true"/></div>

<br/>Computation graph:

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
   -                ones_c -----------
  -                 = 
mask_y              np.ones((num_classes, num_classes))
= 
(np.array([y]).T == np.arange(num_classes))
```

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

From the forward graph, we have:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f5.svg?sanitize=true"/></div>

<br/>Take derivatives and use trace tricks, we get:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f6.svg?sanitize=true"/></div>

<br/>Trus, we finally get the gradient:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f7.svg?sanitize=true"/></div>

<br/>Write it in Numpy form:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f8.svg?sanitize=true"/></div>

<br/>A possible code:

```Python
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

<br/>

## Q3: Implement a Softmax classifier

Like Q2, we use matrix derivative methods to get the vectorlized gradient.

Variables:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f9.svg?sanitize=true"/></div>

<br/>Take derivatives:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f10.svg?sanitize=true"/></div>

<br/>Expand S by W:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f11.svg?sanitize=true"/></div>

<br/>Thus, we have:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f12.svg?sanitize=true"/></div>

<br/>

```Python
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    P = np.exp(X.dot(W))
    P /= np.sum(P, axis=1, keepdims=True)
    Y = (y.reshape(-1, 1) == np.arange(num_classes)).astype(np.float64)
    
    loss = reg * np.sum(np.square(W)) - np.sum(Y * np.log(P)) / num_train
    dS = (P - Y) / num_train
    dW = 2 * reg * W + X.T.dot(dS)
    #dX = dS.dot(W.T)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
```

<br/>

## Q4: Two-Layer Neural Network

It's hard to conclute all the important ideas in this part, so for details of training and tuning the network, please see <a href="https://nbviewer.jupyter.org/github/dashidhy/cs231n_assignments/blob/master/cs231n_assignment1/assignment1/two_layer_net.ipynb" target="_blank">my notebook of this question</a>. Concisely, adding dropout and implementing advanced optimizer may help improve the results.

<br/>

## Q5: Higher Level Representations: Image Features

Quite like Q4, nothing more to say.

---

<br/>


# Assignment 2

## Q1: Fully-connected Neural Network 

Nothing is tricky. Just be careful when coding to keep a right data flow in the network. Proper indents, blank lines and annotations will help a lot.

<br/>

## Q2: Batch Normalization

According to the comment, it's possible to write batchnorm_backward_alt in a single 80-character line. However, though I have tried my best to comepress the computation and my code, I still need 4 lines. Here is my code. It's 0.4 times faster than the non-optimized one.

```Python

def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    gamma, norm, scale = cache
    dbeta = np.sum(dout, 0)
    dgamma = np.sum(dout*norm, 0)
    dcenter = dout*gamma/scale
    dx = dcenter-np.mean(dcenter, 0)-np.mean(dcenter*norm, 0)*norm
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta
```

<br/>

## Q3: Dropout

Easy.

<br/>

## Q4: Convolutional Networks

Just follow the instruction. The fast conv layers are speeded up by Cython back-end and you don't have to care about it.

<br/>

## Q5: PyTorch / Tensorflow on CIFAR-10

I choose PyTorch. It's friendly for beginners to handle the basic functions. The most tricky part is how to design a good network that works well on CIFAR-10. As a beginner it's hard to have much insight of CNN structures and designed your own network. Since it's time consuming to do a lot of experiments, I recommend you to refer to some well developed structures like VGG, GoogLeNet, and ResNet. I build a mini VGG net that stacks 3x3 conv layers and it obtains ~80% accuracy both on validation set (81.8%, 818/1000) and test set (78.58%, 7858/10000). Here is the Pytorch code, optimizer and hyperparameters are picked according to the <a href="https://arxiv.org/pdf/1409.1556.pdf" target="_blank">the original paper of VGG</a>.

```Python
dhy_mini_VGG = nn.Sequential(

    # 1st module
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    
    # 2nd module
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    
    # 3rd module
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    
    Flatten(),
    
    # FC layers
    nn.Linear(2048, 2048),
    nn.ReLU(inplace=True),
    nn.Dropout(inplace=True),
    nn.Linear(2048, 1024),
    nn.ReLU(inplace=True),
    nn.Dropout(inplace=True),
    nn.Linear(1024, 10)
    
)

model = dhy_mini_VGG.type(gpu_dtype)
loss_fn = nn.CrossEntropyLoss().type(gpu_dtype)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
```

---

<br/>

# Assignment 3

This assignment basically help you to learning implementing simple tools. Following the instructions in the notebooks and refering to course slides and official documentations will help you get good results and make progress. If you are confused by how a transposed conv layer works, <a href="https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0" target="_blank">this article may answer your questions</a>.

Bonus parts are waiting to be done.
