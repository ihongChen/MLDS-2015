# coding:utf8

import numpy as np
import theano
import theano.tensor as T
from sklearn.model_selection import train_test_split
from collections import OrderedDict

filepath = '/home/ihong/Dropbox/py/DL/2015MLDS/hw/dataset/MLDS_HW1_RELEASE_v1/'



############### data preprocessing ###############
def datasets(filepath, test_size=0.1):
    whole_data = {}
    ## load mfcc X data , extract 39 features in one frame
    with open(filepath + 'mfcc/train.ark','rb') as f:
        for lines in f:
            frames = lines.split(' ')
            whole_data[frames[0]] = np.reshape([float(x) for x in frames[1:]],(1,39))

    # load label data into dict format (),Y
    label_48_phones = {}
    with open(filepath + 'label/train.lab','rb') as f :
        for lines in f:
            frames = lines.split(',')
            label_48_phones[frames[0]] = frames[1].strip()

    # mapping 48 phones to numbers
    char_to_num = {} # mapping phones to numbers
    mapping_48to39 = {} # mapping 48 phones to 39 (numbers)
    with open(filepath + 'phones/48_39.map') as f:
        for nu,lines in enumerate(f):
            phone48, phone39 = lines.split('\t')
            char_to_num[phone48] = nu

    with open(filepath + 'phones/48_39.map') as f:
        for nu,lines in enumerate(f):
            phone48, phone39 = lines.split('\t')
            phone39 = phone39.rstrip()
            mapping_48to39[nu] = char_to_num[phone39]

    ## data sets ,dataX,dataY
    dataX = [] # data sets for input X
    dataY = [] # data sets for real output value y

    for instanceID,data in whole_data.items():
        dataX.append(data.ravel()) # 2d to 1d
        ylabel = label_48_phones[instanceID] #
        dataY.append(char_to_num[ylabel])

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    X_train,X_test, y_train, y_test = train_test_split(dataX, dataY,\
                                        test_size = test_size, random_state = 4)
    return X_train, X_test, y_train, y_test


#####################################################
## Vanilla Neural Network (Baseline)
#####################################################

## initialize in theano

def init_weight(shape,index):
    return theano.shared(0.01* np.random.randn(*shape),name = 'w_{}'.format(index))

def init_bias(dim,index):
    return theano.shared(np.zeros(dim),name = 'b_{}'.format(index))

######## update strategy ######
### mini-batch GD ###
def gd(cost,params,mu):
    gradients = T.grad(cost,params)
    parameters_update = \
    [(p,p-mu*g) for p,g in zip(params,gradients)]
    return parameters_update
### Adagrad ###
## from :https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py#L385
def adagrad(cost, params, learning_rate=1.0, epsilon=1e-6):
    """Adagrad updates
    Scale learning rates by dividing with the square root of accumulated
    squared gradients. See [1]_ for further description.
    Parameters
    ----------
    cost : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    Using step size learning_rate Adagrad calculates the learning rate for feature i at
    time step t as:
    .. math:: \\learning_rate_{t,i} = \\frac{\\learning_rate}
       {\\sqrt{\\sum^t_{t^\\prime} g^2_{t^\\prime,i}+\\epsilon}} g_{t,i}
    as such the learning rate is monotonically decreasing.
    Epsilon is not included in the typical formula, see [2]_.
    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
           Adaptive subgradient methods for online learning and stochastic
           optimization. JMLR, 12:2121-2159.
    .. [2] Chris Dyer:
           Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    """
    grads = T.grad(cost,params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))
    return updates

### momentum ###

######## main #########################

if __name__ == '__main__':
    # parameters setup
    X = T.matrix('X')
    y = T.lvector('y')

    nn_input_dim = 39 # 39 features as input dim
    nn_output_dim = 48 # 48 categorical phones
    hidden_shape = [128,128] # shape of hidden layer
    nn_shape = [nn_input_dim] + hidden_shape + [nn_output_dim] # [39,1000,1000,48] nn shape


    ### initialize weight/bias ###
    W = {};b = {} # weight and bias
    layers = range(len(nn_shape) - 1) # [0,1,2]
    for layer in layers :
        shape = nn_shape[layer:layer+2] # shape of this layer
        dim = nn_shape[layer+1] # dim of bias
        W[layer] = init_weight(shape,index=layer)
        b[layer] = init_bias(dim,index=layer)

    ### feed-foward
    temp = X ; z = {}; a = {};
    layers = range(len(nn_shape) - 1) # [0,1,2]
    learning_rate = 0.1 # learning rate
    params = [W[i] for i in layers] + [b[i] for i in layers]

    for layer in layers: #layers:[0,1,2]
        z[layer] = temp.dot(W[layer]) + b[layer]
        if layer == layers[-1]:
            break
        a[layer] = T.nnet.relu(z[layer]) # ReLu
        temp = a[layer]

    yhat = T.nnet.softmax(z[layer])
    foward_prop = theano.function([X],yhat) ## foward_prop

    ### cost(loss) function
    cost = T.nnet.categorical_crossentropy(yhat,y).mean()
    calculate_cost = theano.function([X,y],cost) # function
    ### update
    # gradient_step = theano.function(
    #                 [X,y],updates = gd(cost,params,mu=learning_rate))
    gradient_step = theano.function([X,y],updates = adagrad(cost,params,learning_rate=learning_rate))
    ### predict
    prediction = T.argmax(yhat, axis=1)
    predict = theano.function([X],prediction)

    ### split data, train/test
    X_train,X_test,y_train,y_test = datasets(filepath,test_size=0.1)

    batch_size = 128
    epochs = 100
    print '''nn_shape:{},\nbatch_size:{},\nlearning rate:{},\nsample_size: {}\n'''.format(nn_shape,batch_size,learning_rate,len(X_train))
    for epoch in range(epochs):

        ## shuffle the training sets
        temp = zip(*(X_train,y_train))
        np.random.shuffle(temp)
        X_train, y_train = zip(*temp)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        n = len(X_train) # size of training set

        X_mini_batches = [X_train[k:k + batch_size] for k in xrange(0,n,batch_size)]
        y_mini_batches = [y_train[k:k+batch_size] for k in xrange(0,n,batch_size)]
        loss = 0
        for X_mini_batch,y_mini_batch in zip(X_mini_batches,y_mini_batches):
            gradient_step(X_mini_batch,y_mini_batch)
            loss += calculate_cost(X_mini_batch,y_mini_batch)
        loss = loss/len(X_mini_batches)
        accuracy = float(sum(predict(X_test) == y_test)) / len(X_test) # accuracy
        print "loss after epoch:{epoch}, loss:{loss}, accuracy:{acc} ".format(\
            epoch = epoch+1, loss = loss, acc = accuracy)

#######
