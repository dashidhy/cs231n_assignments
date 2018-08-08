# Some tips of cs231n spring 2017 assignments

Here I take some notes about some of my ideas that I think may be useful in my future programming.

---

# Assignment 1

## Q1: k-Nearest Neighbor Classifier

In the KNN problem, the most tricky part is how to fully vectorlize the computation of l2 distance. The solution is that first, **decompose the final distance matrix,** and second, **vectorlize the computing process with matrix multiplication and broadcast sums.**

Consider each element of the final distance matrix: 

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f1.svg?sanitize=true"/></div>

<br/>In this way, we can decompose each element of the dist matrix into three terms. Each of the terms can be vectorlized into a Numpy form:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f2.svg?sanitize=true"/></div>

<br/>Thus, the fully vectorlized L2 distance code is something like the code below:

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

Trus, we finally get the gradient:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f7.svg?sanitize=true"/></div>

<br/>Write it in Numpy form:

<div align=center><img src="https://github.com/dashidhy/cs231n_assignments/raw/master/figure/f8.svg?sanitize=true"/></div>

<br/>A possible original code is the one below:

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

## Q3: Implement a Softmax classifier

Here I give my idea of how to reach a fully vectorized Softmax classifier. Note that the code is actually from Q4, which also gives ideas of the forward and backward data flow a simple fully connected neural network.

```Python
def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    C = b2.shape[0]

    # Compute the forward pass
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    s1 = X.dot(W1)
    f1 = s1+b1
    f1_0 = f1 > 0
    f1_relu = f1*f1_0
    s2 = f1_relu.dot(W2)
    scores = s2+b2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    S = np.exp(scores)
    S /= np.array([np.sum(S, 1)]).T
    loss = reg*(np.sum(W1 * W1)+np.sum(W2 * W2))-np.sum(np.log(S[list(range(N)), y]))/N
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    ##### BP structures #########################################################
    #############################################################################
    # Matrix multiplication:                                                    #
    #                                                                           #
    #                 <---- dA = dC.dot(B.T)                                    #
    #               A -----------                                               #
    #                            -                                              #
    #                             -                 <---- dC = ...              #
    #                              --> C = A.dot(B) ----------->                #
    #                             -                                             #
    #                            -                                              #
    #               B -----------                                               #
    #                 <---- dB = A.T.dot(dC)                                    #
    #                                                                           #
    #############################################################################
    # Bias plus (including broadcast):                                          #
    #                                                                           #
    #                      <-- dS = df                                          #
    #        S = X.dot(W) -----------                                           #
    #                                 -                                         #
    #                                  -            <-- df = ...                # 
    #                                   --> f = S+b ----------->                #
    #                                  -                                        #
    #                                 -                                         #
    #                    b -----------                                          #
    #                      <-- db = np.sum(df, 0)                               #
    #                                                                           #
    #############################################################################
    # ReLU:                                                                     #
    #                                                                           #
    #               <-- df = (f > 0)*df_relu      <-- df_relu = ...             #
    #             f -----------> f_reku = ReLU(f) ----------->                  #
    #                                                                           #
    #############################################################################
    # Softmax loss:                                                             #
    #                                                                           #
    # The gradient of Softmax loss is a little bit hard to compute. It may be-  #
    # come easier to compute it by decomposing the process to several basic     #
    # structures.                                                               #
    #___________________________________________________________________________#
    # 1. Exponent                                                               #
    #                                                                           #
    #                 <-- df = Se*dSe             <-- dSe = ...                 #
    #               f -----------> Se = np.exp(f) ----------->                  #
    #___________________________________________________________________________#
    # 2. Normalization through the 1st dimention                                #
    #                                                                           #
    # The normalization process can be further decomposed into several basic    #
    # computations.                                                             #
    #                                                                           #
    #      <-- dSe = E*dS                                        <--dS = ...    #
    #   Se -------------------------------------------> S = Se*E ---------->    #                            -
    #     -                                        -                            #
    #      - <-- dSe = dD.dot(C.T)                -                             #
    #       -                                    -                              #
    #        -                                  - <-- dE = Se*dS                #
    #         -                                -                                #
    #          --> D = Se.dot(C) -----------> E = 1/D                           #
    #         -                  <-- dD = -(1/(D*D))*dE                         #
    #        -                                                                  #
    #       -                                                                   #
    #      -                                                                    #
    #     -                                                                     #
    #   C = np.ones(Se.shape[1], Se.shape[1])                                   #
    #                                                                           #
    # Then we can merge the computation to a relatively simple form:            #
    #                                                                           #
    #                dSe = (1/D)*dS-((1/(D*D))*Se*dS).dot(C.T)                  #
    #___________________________________________________________________________#
    # 3. Logarithm                                                              #
    #                                                                           #
    #                <-- dS = (1/S)*dSl          <-- dSl = ...                  #
    #              S -----------> Sl = np.log(S) ----------->                   #
    #___________________________________________________________________________#
    # 4. Choose the scores of correct categories and then compute the loss      #
    #                                                                           #
    #    <-- dSl = -cr/num_train                                                #
    # Sl -----------                                                            #
    #               -                                                           #
    #                -               <-- dSc = -np.ones(Sc.shape)/num_train     #
    #                 --> Sc = Sl*cr -----------> loss = -np.sum(Sc)/num_train  #
    #                -                                                          #
    #               -                                                           #
    #             cr = (np.array([y]).T == np.arange(num_classes))              #
    #___________________________________________________________________________#
    # 5. Merge the computations                                                 #
    #                                                                           #
    # Finally, we can merge the computations above to reach a simple form of    #
    # the gradient of the Softmax loss. Note that S = Se*(1/D) we have:         #
    #                                                                           #
    #       dS = -(1/S)*(cr/num_train)                                          #
    #                                                                           #
    # ==>                                                                       #
    #                                                                           #
    #      dSe = -(1/D)*(1/S)*(cr/num_train)                                    #
    #            +((1/(D*D))*Se*(1/S)*(cr/num_train)).dot(C.T)                  # 
    #                                                                           #
    #          = -(1/Se)*(cr/num_train)+((1/D)*(cr/num_train)).dot(C.T)         #
    #                                                                           #
    #          = -(1/Se)*(cr/num_train)+(1/D)/num_train                         #
    #                                                                           #
    # ==>                                                                       #
    #                                                                           #
    #       df = Se*dSe                                                         #
    #                                                                           #
    #          = (Se/D-cr)/num_train                                            #
    #                                                                           #
    #          = (S-cr)/num_train                                               #
    #                                                                           #
    #############################################################################
    ds2 = (S-(np.array([y]).T == np.arange(C)))/N
    db2 = np.sum(ds2, 0)
    dW2 = 2*reg*W2+(f1_relu.T).dot(ds2)
    df1 = f1_0*(ds2.dot(W2.T))
    db1 = np.sum(df1, 0)
    dW1 = 2*reg*W1+(X.T).dot(df1)
      
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads
```

## Q4: Two-Layer Neural Network

It's hard to conclute all the important ideas in this part, so for details of training and tuning the network, please see <a href="https://nbviewer.jupyter.org/github/dashidhy/cs231n_assignments/blob/master/cs231n_assignment1/assignment1/two_layer_net.ipynb" target="_blank">my notebook of this question</a>. Concisely, adding dropout and implementing advanced optimizer may help improve the results.



## Q5: Higher Level Representations: Image Features

Quite like Q4, nothing more to say.

---

# Assignment 2

## Q1: Fully-connected Neural Network 

Nothing is tricky. Just be careful when coding to keep a right data flow in the network. Proper indents, blank lines and annotations will help a lot.

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

## Q3: Dropout

Easy.

## Q4: Convolutional Networks

Just follow the instruction. The fast conv layers are speeded up by Cython back-end and you don't have to care about it.

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

# Assignment 3

This assignment basically help you to learning implementing simple tools. Following the instructions in the notebooks and refering to course slides and official documentations will help you get good results and make progress. If you are confused by how a transposed conv layer works, <a href="https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0" target="_blank">this article may answer your questions</a>.

Bonus parts are waiting to be done.
