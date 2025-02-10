from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    #     # here, it is easy to run into numeric instability. Don't forget the        #
    #     # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    for i in range(num_train):
        score = np.dot(X[i], W)
        score = np.exp(score)
        score = score / np.sum(score)

        X_temp = np.repeat(X[i].reshape(-1, 1), 10, axis=1)
        # 10번 반복 했으므로, 3072x10이 된다.

        dW += X_temp * score / num_train  # 3072x10
        dW[:, y[i]] += -X[i] / num_train
        # softmax_gradient는 정답 class는 실제 확률보다 더 큰 영향을 줘야 하기에 -x[i]를
        # 추가적으로 빼준다

        loss += -np.log(score[y[i]])

    dW += 2 * reg * W

    loss /= num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    scores = np.dot(X, W)
    exp_scores = np.exp(scores)
    prob_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_log_probs = -np.log(prob_scores[range(num_train), y])
    # 전체적인 과정은 naive와 비슷하지만 한번에 dot product를 연산 처리하고
    # loop없이 연산을 진행한다.

    loss = np.sum(correct_log_probs)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W ** 2)

    # grads
    dscores = prob_scores
    dscores[range(num_train), y] -= 1
    # 정답 class에서의 중요성을 두기위해 -1을 뺴줘서 backprop을 진행한다.
    dW = np.dot(X.T, dscores)
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
