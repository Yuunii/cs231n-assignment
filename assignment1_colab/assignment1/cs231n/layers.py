from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    row_dim = x.shape[0]
    col_dim = np.prod(x.shape[1:])
    x_reshape = x.reshape(row_dim, col_dim)
    out = np.dot(x_reshape, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M) # M은 이 층의 출력 뉴런의 수를 의미한다.
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = cache[0]
    w = cache[1]
    b = cache[2]

    raw = x.shape[0]
    col = np.prod(x.shape[1:]) # x의 shape가 (N, 3, 32, 32)라고 하면 x.shape[:1]은 (3,32,32)가 된다.
    # 즉 3072이가 되므로, d가 된다.
    x_reshape = x.reshape(raw, col)
    dw = x_reshape.T.dot(dout) # mul gate를 생각하면 dw를 구할 때는 x에 대한 식만 남고
    dx = dout.dot(w.T).reshape(x.shape) # dx를 구할 때는 w에 대한 식만 남는다.

    db = np.sum(dout, axis=0)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    relu = np.maximum(0,x)
    # np.maximum은 두 array의 각 원소의 크기를 비교하여 값을 반영하는 메소드
    # np.max는 특정 축에 대하여 최댓값을 반영하는 메소드


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    relu = np.maximum(0,x)
    if relu>0:
        relu = 1
    dx = dout * relu

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        # men


        sample_mean = np.mean(x, axis =0)
        sample_var = np.mean(x, axis=0)
        x_hat = (x-sample_mean) / np.sqrt(sample_var)
        out = x_hat * gamma + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = {}
        cache['sample_mean'] = sample_mean
        cache['sample_var'] = sample_var
        cache['x_hat'] = x_hat
        cache['x'] = x
        cache['gamma'] = gamma
        cache['beta'] = beta
        cache['eps'] = eps


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_hat = (x - running_mean) / np.sqrt(running_var)
        out = x_hat * gamma + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    m = dout.shape[0]


    # dout = gamma * x_hat + beta의 식을 각각 편미분을 해야된다.
    # 즉 dout = gamma * (x-mean / (var + eps) ** 0.5 ) + beta가 되고
    # mean = np.mean(x, axis=0 ) , var = np.maen((x-mean)**2, axis =0)가 되고
    # dvar= = dl / dout * dout / dvar
    # dx = dl/dout * dout/dx
    # dgamma = dl/dout * x_hat
    # dbeta = dl/dout * 1


    #이와 같이 각각 상황에 맞게 backprop을 진행하면 다음과 같은 코드가 진행된다.

    dx_hat = dout * cache['gamma']
    dsample_var = np.sum(dx_hat * (cache['x'] - cache['sample_mean']) * (-0.5) * (cache['sample_var'] + cache['eps']) ** (-1.5), axis=0)


    dsample_mean = (np.sum(dx_hat * (-1 / np.sqrt(cache['sample_var'] + cache['eps'])), axis=0) +
                     dsample_var * ((np.sum(-2 * (cache['x'] - cache['sample_mean']))) / m))

    dx = dx_hat * (1 / np.sqrt(cache['sample_var'] + cache['eps'])) + \
         dsample_var * (2 * (cache['x'] - cache['sample_mean']) / m) + \
         dsample_mean / m

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * cache['x_hat'], axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # batch norm과 계산과정은 크게 다를 건 없지만,
    # 모든 채널을 합하여서 평균을 내므로 axis=1이어야 한다.

    feature_mean = np.mean(x, axis =1)
    feature_var = np.var(x, axis=1)

    x_hat = (x-feature_mean) / np.sqrt(feature_var + eps)
    out = gamma * x_hat + beta

    cache = (x, feature_mean, feature_var, gamma)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    # Batch norm과 다르지 않게 layer norm도 상황에 맞게 미분을 해서
    # 구하면 되고, batch norm과 전체적인 흐름이 비슷하다.

    eps = 1e-5
    N = dout.shape[1]
    x, feature_mean, feature_var, gamma = cache
    x_normal = (x - feature_mean) / np.sqrt(feature_var + eps)
    dgamma = np.sum(dout * x_normal, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_normal = dout * gamma

    dlvar = np.sum(dx_normal * (x - feature_mean)
                   * -0.5 * (feature_var + eps) ** -1.5, axis=1)[:, np.newaxis]

    dlmean = (np.sum(dx_normal * -1 / np.sqrt(feature_var + eps), axis=1)[:, np.newaxis] +
              dlvar * np.sum(-2 * (x - feature_mean), axis=1)[:, np.newaxis] / N)

    dx = dx_normal * 1 / np.sqrt(feature_var + eps) + dlvar * 2 * (x - feature_mean) / N + dlmean / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p
        # 여기서 p를 나눠주는 이유는 dropout을 통해 잃어버린 값 만큼
        # p의 역수만큼 곱해줘 scaling을 해준다.

        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # dropout은 test시에는 진행하지 않기에 그냥 x값을 넘겨주면 된다.
       out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, CC, HH, WW = w.shape
    assert C == CC

    H_out = 1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride']
    W_out = 1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride']

    out = np.zeros((N, F, H_out, W_out))

    # padding
    pad = conv_param['pad']
    x_with_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    # padding은 N,C에선 적용하지 않고
    # H,W에만 padding을 적용한다.
    _, _, H, W = x_with_pad.shape

    # convolving
    stride = conv_param['stride']

    for i in range(0, N): # 전체 데이터 샘플 횟수
        x_data = x_with_pad[i]

        xx, yy = -1, -1 # row, column 이동횟수 count
        for j in range(0, H - HH + 1, stride):
            yy += 1
            for k in range(0, W - WW + 1, stride):
                xx += 1
                x_rf = x_data[:, j:j + HH, k:k + WW]

                for l in range(0, F):
                    conv_value = np.sum(x_rf * w[l]) + b[l] # dot product
                    out[i, l, yy, xx] = conv_value

            xx = -1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    pad = conv_param['pad']
    stride = conv_param['stride']
    x_with_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    N, F, Hdout, Wdout = dout.shape
    # 출력 feature map의 크기


    db = np.zeros((b.shape))
    for i in range(0, F):
        db[i] = np.sum(dout[:, i, :, :])


    dw = np.zeros((F, C, HH, WW))
    for i in range(0, F): # filter 개수
        for j in range(0, C): # input channel 개수
            for k in range(0, HH): # kernel height
                for l in range(0, WW): # kernel width
                    dw[i, j, k, l] = np.sum(
                        dout[:, i, :, :] * x_with_pad[:, j, k : k + Hdout * stride : stride, l : l + Wdout * stride : stride])
                    # dout의 i번째 filter를 가져오고, x_with_pad에서는 j번째 channel에서
                    # k~k+ Hout * strid, l ~ l + Wdout * stride
                    # input data에서 필터가 적용된 모든 위치를 stride 간격으로 선택하는 역할
                    # dout[n,f,i,j] * x_with_pad[n,c, i* stride + k, j * stirde + l]
                    # x_with_pad는 가중치가 적용된 영역

    dx = np.zeros((N, C, H, W))
    for nprime in range(N): # 데이터 샘플마다 수행
        for i in range(H): # height
            for j in range(W): # width
                for f in range(F): # filter 개수별
                    for k in range(Hdout): # 출력 feature map 높이
                        for l in range(Wdout): # 출력 feature map 너비
                            mask1 = np.zeros_like(w[f, :, :, :])
                            mask2 = np.zeros_like(w[f, :, :, :])
                            # Conv의 filter는 input x의 특정 영역과만 연산을 수행한다.
                            # 해당 영역을 mask를 통해 선택적으로 활성화를 하고 누적 시켜야 backprop이 된다.

                            if (i + pad - k * stride) < HH and (i + pad - k * stride) >= 0:
                                mask1[:, i + pad - k * stride, :] = 1.0
                            if (j + pad - l * stride) < WW and (j + pad - l * stride) >= 0:
                                mask2[:, :, j + pad - l * stride] = 1.0
                            # 다음과 같은 식은 input image의 특정 픽셀 (i,j)에 대하여 filter 내부에 존재하는 지 확인하고
                            # 있으면 1로 marking 하는 역할을 한다.

                            w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                            # channel과 높이 방향으로 더해진다.
                            # 즉 WW만 남게 된다.
                            dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked
                            # dl/dx = dout * w


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    pool_stride = pool_param['stride']

    H_out = 1 + (H-pool_height) / pool_stride
    w_out = 1 + (W-pool_width) / pool_stride

    out = np.zero(N,C,pool_height, pool_width)


    #conv_forward와 유사하다.
    for i in range(0,N):
        x_data = x[i]

        xx, yy = -1
        for j in range(0,H-pool_height+1, pool_stride):
            yy+=1
            for k in range(0, W-pool_width+1, pool_stride):
                xx+=1
                x_rf = x_data[:,j:j+pool_height,k:k+pool_width]
                for c in range(0,C):
                    out[i,j,k,c] = np.max(x_rf[c])

                xx = -1



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache




def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N,C,H,W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    pool_stride = pool_param['stride']

    dx = np.zero((N,C,H,W))
    H_out = 1 + (H - pool_height) / pool_stride
    w_out = 1 + (W - pool_width) / pool_stride


    for i in range(0,N):
        x_data = x[i]

        xx, yy = -1
        for j in range(0,H-pool_height+1, pool_stride):
            yy+=1
            for k in range(0, W-pool_width+1, pool_stride):
                xx+=1
                x_rf = x_data[:,j:j+pool_height,k:k+pool_width]
                for c in range(0,C):
                    x_pool = x_rf[c]
                    if x_pool == np.max(x_rf[c]):
                        # maxpool 자체가 큰 값만 넘겨주면 되기에
                        # np.max를
                        mask = x_pool

                    dx[i,c,j:j+pool_height,k:k+pool_width] += dout[i,c,xx,yy] * mask



                xx = -1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x_reshape = x.transpose(0,2,3,1).reshape(N*H*W,C) # N,C,H,W -> N,H,W,C
    out_tmp, cache = batchnorm_forward(x_reshape,gamma, beta, bn_param)
    out = out_tmp.reshape(N, H, W, C).transpose(0, 3, 1, 2) # N,H,C,W -> N,C,H,W 가 된다.

    # 여기서 transpose를 하는 이유는 해당하는 channel 내에 mini batch들이
    # spatial space를 무시한 채 다 섞여버려 같은 channel 내에 있는 feature들이 하나로 묶여버림
    # 그래서 transpose를 거친 뒤에 reshape를 진행해야 spatial를 유지한 채 Batch_norm을 진행할 수 있다.




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = dout.shape
    dout_reshape = dout.transpose(0,2,3,1).reshape(N*H*W,C)
    dx_tmp, dgamma, dbeta = batchnorm_backward(dout_reshape)
    dx = dx_tmp.reshape(N,C,H,W).transpose(0,3,1,2)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = x.shape
    x_reshape = x.reshape([N,G,C//G,H,W])
    # group norm은 C개의 channel을 G개로 묶어서 Normalization을 진행한다.
    # 그래서 G는 Group의 갯수, C//G는 하나의 그룹에서 가지는 channel의 개수가 된다.

    x_mean = np.mean(x_reshape, axis=(2, 3, 4), keepdims=True)
    x_var = np.var(x_reshape, axis=(2,3,4),keepdims=True)
    x_std = np.sqrt(x_var + eps)
    # 여기서 axis(2,3,4)는 C//G, H,W를 택해서 mean과 variance를 계산한 것이고
    # keepdims가 False일때 dim은 (N,G)가 되어 Broadcasting이 불가능하기에
    # keepdims를 True로 설정하여 (N,G,1,1,1)이 가능하게 된다.

    x_hat = (x- x_mean) / x_std
    out = gamma * x_hat + beta
    cache = x, x_mean, x_std, x_var, x_hat, gamma, beta


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 여기서도 batch norm, layer norm과 연산과정의 전체적인 흐름에 있어서는 크게 다를 것이 없다.
    # 하지만 batch norm의 평균을 낼 땐 axis = 0
    # layer norm의 평균을 낼 땐 axis = 1
    # group norm에서는


    x, x_mean, x_var, x_std, gamma, x_hat, G = cache
    N, C, H, W = dout.shape
    xg = x.reshape([N, G, C // G, H, W])
    M = C // G * H * W
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3)).reshape(1, C, 1, 1)
    dbeta = np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1)
    # 채널별로 곱한 후에 그 값을 합산한 결과를 저장
    # N, H, W 차원에 맞게 broadcasting을 할 수 있도록 차원을 맞추는 작업을
    # .reshape(1,C,1,1)을 통해 해준다.
    # keepdims = True로 설정해도 된다.


    # axis=(2,3,4)를 통해 C // G, H, W의 평균을 구해서
    # batch norm, layer norm 처럼 gradient를 구하면 된다.
    dvar = np.sum((dout * gamma).reshape(xg.shape) * (xg - x_mean) * (-0.5) * (x_std ** (-1.5)), axis=(2, 3, 4),
                  keepdims=True)
    du = np.sum(((-dout * gamma).reshape(xg.shape) / x_std), axis=(2, 3, 4), keepdims=True) + dvar * np.sum(
        (-2 * (xg - x_mean)), axis=(2, 3, 4), keepdims=True) / M
    dx = (dout * gamma).reshape(xg.shape) / x_std + dvar * 2 * (xg - x_mean) / M + du / M
    dx = dx.reshape(N, C, H, W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    """
    
    Assignment1에서 했습니다.
    
    """

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    """
    
    Assignment1에서 했습니다.
    
    """


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
