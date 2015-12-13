
import sys
import numpy as np
from math import sqrt
from math import log
from scipy.optimize import minimize

def encodeMatrix(inputfile):
    d = {}
    count = []

    temp = [line.rstrip("\n").split("\t") for line in open(inputfile, "rb")]


    for i in range(1, len(temp)):
        for k in range(1, len(temp[i])):
            count.append(0)
            try:
                temp[i][k] = float(temp[i][k])
            except ValueError:
                temp[i][k] = float(getOneHotEncoding(d, temp[i][k], count, k))
                continue
    return temp

def getOneHotEncoding(d, name, count, columnNo):
    if columnNo not in d :
        x = {}
        x[name] = 0
        d[columnNo] = x
        count[columnNo] = 0
        return 0
    else :
        x = d[columnNo]
        if name not in x :
            c = count[columnNo]
            c = c + 1
            count[columnNo] = c
            x[name] = c
            return c
        else :
            c = x[name]
            return c

def preprocess(inputfile,testfile):
    mat1 = encodeMatrix(inputfile)
    mat2 = encodeMatrix(testfile)
    mat1 = np.array(mat1, dtype=float)
    mat2 = np.array(mat2, dtype=float)
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
    n_training = len(mat1)
    n_validation = round(len(mat1) * 0.15)
    n_test = len(mat1) - n_training

    t1 = range(len(mat1))  # Shuffling the rows of testmat
    aperm = np.random.permutation(t1)
    labels = np.array(mat1[:,0])
    datamat = np.array(mat1[:, 1:mat1.shape[1]])

    t2 = range(len(mat2))  # Shuffling the rows of testmat
    aperm = np.random.permutation(t2)
    test_label = np.array(mat2[:,0])
    test_data = np.array(mat2[:, 1:mat2.shape[1]])
    test_data = test_data / 255.0
    # normalize!!
    data = datamat / 255

    train_data = np.array(data[aperm[0:n_training], :])
    train_label = np.array(labels[aperm[0:n_training]])
    validation_data = np.array(data[aperm[n_validation:len(data)], :])
    validation_label = np.array(labels[aperm[n_validation:len(labels)]])
    #test_data = np.array(data[aperm[n_test:len(data)], :])
    #test_label = np.array(labels[aperm[n_test:len(labels)]])
    return train_data, train_label, validation_data, validation_label, test_data , test_label

def initializeWeights(n_in, n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon;
    return W

def sigmoid(z):
    x = np.divide(1.0, (1.0 + np.exp(-z)))
    return  x

def nnObjFunction(params, *args):
    n_input, hidden, n_class, training_data, training_label, lambdaval = args

    #level 1 matrix
    w1 = params[0:hidden * (n_input + 1)].reshape((hidden, (n_input + 1)))  # check this whty its coming i*10
    # w1=np.divide(w1,10)

    #level 2 matrix
    w2 = params[(hidden * (n_input + 1)):].reshape((n_class, (hidden + 1)))
    obj_val = 0
    train_label = np.array([])
    # initialize training label to [0,0,0,0]
    train_label = np.zeros((len(training_label), 4))
    # to make [1]  to [0,0,1,0] and 2 to [0,2,0,0]
    for i in range(len(training_label)):
        train_label[i][training_label[i]] = 1
    #add bias node to input
    training_data = np.append(training_data, np.ones([len(training_data), 1]), 1)

    # We calculate the output of hidden node

    aj = np.dot(training_data, np.transpose(w1))
    zj = sigmoid(aj)
    # add bias to the output of hidden node
    zj = np.append(zj, np.ones([len(zj), 1]), 1)

    # Calculate the output
    bl = np.dot(zj, w2.T)
    ol = sigmoid(bl)
    # deltal is (actual-target)
    deltaL = ol - train_label

    #(actual-target)* outputof hidden node
    product_deltaL_zj = np.dot(deltaL.T, zj)

    # w2*lambda
    product_lambda_w2 = np.multiply(lambdaval, w2)

    # (actual-target)* (output of hidden node)* lamda *w2
    grad2 = (product_deltaL_zj + product_lambda_w2)

    # we divided the above with length of training data
    grad2 = np.divide(grad2, len(training_data))

    # deltaL*W2
    C = np.dot(deltaL, w2)

    X = np.ones(zj.shape)

    ABT = (1 - zj) * (zj) * C

    ABT = np.delete(ABT, np.divide(ABT.size, len(ABT)) - 1, 1)
    ABTCD = np.dot(ABT.T, training_data)
    product_lambda_w1 = np.multiply(lambdaval, w1)
    grad1 = ABTCD + product_lambda_w1
    grad1 = np.divide(grad1, len(training_data))
    sum = 0
    len1 = len(train_label)
    len2 = len(training_data)
    s1 = train_label.size / len(train_label)
    for i in range(len(train_label)):
        for j in range(int(round(s1))):
            if(train_label[i][j] == 1):
                sum = sum + log(ol[i][j])
            else:
                sum = sum + log((1 - ol[i][j]))
    sum = np.divide(sum, len(training_data))
    sum = np.multiply(sum, -1)
    w1square = np.power(w1, 2)
    w2square = np.power(w2, 2)
    w1flat = w1square.flatten()
    w1sum = np.sum(w1flat)
    w2flat = w2square.flatten()
    w2sum = np.sum(w2flat)
    wsum = w1sum + w2sum

    sum2 = np.multiply(2, len(training_data))
    sum2 = np.divide(lambdaval, sum2)
    sum2 = np.multiply(sum2, wsum)
    obj_val = sum + sum2
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad1.flatten(), grad2.flatten()), 0)
    return (obj_val, obj_grad)

def nnPredict(w1, w2, data):
    labels = np.array([])
    data = np.append(data, np.ones([len(data), 1]), 1)
    aj = np.dot(data, np.transpose(w1))
    zj = sigmoid(aj)
    zj = np.append(zj, np.ones([len(zj), 1]), 1)
    bl = np.dot(zj, w2.T)
    ol = sigmoid(bl)
    labels = np.argmax(ol, 1)
    return labels

def main(argv):
    print(argv)
    inputfile = ''
    hidden = 50
    maxiter = 30
    lambdaval = 0.6
    listArgs = []
    listArgs = argv
    inputfile = listArgs[1]
    hidden = int(listArgs[2])
    maxiter = int(listArgs[3])
    lambdaval = float(listArgs[4])
    testfile = listArgs[5]
    opts = {'maxiter' : maxiter}

    train_data, train_label, validation_data, validation_label,  test_data, test_label  = preprocess(inputfile,testfile);
    n_input = train_data.shape[1]
    n_class = 4;
    initial_w1 = initializeWeights(n_input, hidden);
    initial_w2 = initializeWeights(hidden, n_class);
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
    args = (n_input, hidden, n_class, train_data, train_label, lambdaval)
    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    w1 = nn_params.x[0:hidden * (n_input + 1)].reshape((hidden, (n_input + 1)))
    w2 = nn_params.x[(hidden * (n_input + 1)):].reshape((n_class, (hidden + 1)))
    predicted_label = nnPredict(w1, w2, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))))
    #predicted_label = nnPredict(w1, w2, validation_data)
    #print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))))
    predicted_label = nnPredict(w1, w2, test_data)
    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))))

if __name__ == "__main__":
    main(sys.argv)