from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

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

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      Sample_Index = np.random.choice(num_train, batch_size)
      X_batch = X[Sample_Index, :]
      y_batch = y[Sample_Index]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate*grads['W1']
      self.params['b1'] -= learning_rate*grads['b1']
      self.params['W2'] -= learning_rate*grads['W2']
      self.params['b2'] -= learning_rate*grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    f1 = X.dot(W1)+b1
    f1_relu = f1*(f1 > 0)
    y_pred = np.argmax(f1_relu.dot(W2)+b2, 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

  def loss_dropout(self, X, y=None, reg=0.0, dropout=0.5):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network  with dropout.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.
    - dropout: The rate of dropout.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    C = b2.shape[0]
    
    s1 = X.dot(W1)
    f1 = s1+b1
    
    # Add dropout
    f1_0 = (f1 > 0)*(np.random.rand(*f1.shape) < dropout)/dropout
    
    f1_relu = f1*f1_0
    s2 = f1_relu.dot(W2)
    
    scores = s2+b2
    
    if y is None:
      return scores
    
    S = np.exp(scores)
    S /= np.array([np.sum(S, 1)]).T
    loss = reg*(np.sum(W1 * W1)+np.sum(W2 * W2))-np.sum(np.log(S[list(range(N)), y]))/N
    
    grads = {}
    
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
    
    return loss, grads
 
  def train_dropout(self, X, y, X_val, y_val,
                    learning_rate=1e-3, learning_rate_decay=0.95,
                    reg=5e-6, num_iters=100,
                    batch_size=200, dropout=0.5, verbose=False):
    """
    Train the network using dropout and SGD.
    
    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - dropout: The rate of dropout.
    - verbose: boolean; if true print progress during optimization.
    """
    
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      Sample_Index = np.random.choice(num_train, batch_size)
      X_batch = X[Sample_Index, :]
      y_batch = y[Sample_Index]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss_dropout(X_batch, y=y_batch, reg=reg, dropout=dropout)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate*grads['W1']
      self.params['b1'] -= learning_rate*grads['b1']
      self.params['W2'] -= learning_rate*grads['W2']
      self.params['b2'] -= learning_rate*grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def train_dropout_momentum(self, X, y, X_val, y_val,
                             learning_rate=1e-3, learning_rate_decay=0.95, mu=0.9,
                             reg=5e-6, num_iters=100,
                             batch_size=200, dropout=0.5, verbose=False):
    """
    Train the network using dropout and SGD with momentum.
    
    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - mu: The momentum.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - dropout: The rate of dropout.
    - verbose: boolean; if true print progress during optimization.
    """
    
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # Initialize the velocity
    v_W1 = 0.0
    v_b1 = 0.0
    v_W2 = 0.0
    v_b2 = 0.0
    
    for it in xrange(num_iters):
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      Sample_Index = np.random.choice(num_train, batch_size)
      X_batch = X[Sample_Index, :]
      y_batch = y[Sample_Index]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss_dropout(X_batch, y=y_batch, reg=reg, dropout=dropout)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      
      # Update velocity
      v_W1 = mu*v_W1-learning_rate*grads['W1']
      v_b1 = mu*v_b1-learning_rate*grads['b1']
      v_W2 = mu*v_W2-learning_rate*grads['W2']
      v_b2 = mu*v_b2-learning_rate*grads['b2']
      
      # Update parameters
      self.params['W1'] += v_W1
      self.params['b1'] += v_b1
      self.params['W2'] += v_W2
      self.params['b2'] += v_b2
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }
    
