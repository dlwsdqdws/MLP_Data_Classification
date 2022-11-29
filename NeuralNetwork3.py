import sys
import numpy as np

# ref: http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
def softmax(x):
    x -= x.max()
    num = np.exp(x)
    den = np.sum(num,axis=0)
    return num / den

def d_softmax(x):
    x -= x.max()
    num = np.exp(x)
    return num / np.sum(num, axis=0) * (1- num / np.sum(num, axis=0))

# ref: https://medium.com/@cmukesh8688/activation-functions-sigmoid-tanh-relu-leaky-relu-softmax-50d3778dcea5
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def d_tanh(x):
    return 1 - (tanh(x) ** 2)

# backforward error function : (A[-1] - y) * softmax derivative(Y)
# error = (output - expected) * transfer_derivative(output)
# ref: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
def calc_err(X, Y, label):
    return (X - label) * d_softmax(Y)

if __name__ == "__main__":

    train_data = np.genfromtxt(str(sys.argv[1]), delimiter=',') 
    train_label = np.genfromtxt(str(sys.argv[2]), delimiter=',')
    test_data = np.genfromtxt(str(sys.argv[3]), delimiter=',') 

    # one-hot
    init = np.zeros((len(train_label),2))
    for i in range(len(train_label)):
        init[i][int(train_label[i])] = 1
    train_label = init

    learning_rate = 0.03
    epochs = 220

    # Xavier / Glorot initialization
    # For tanh, upper and lower limits are Uniform distribution of sqrt(6/(node_in+node_out))
    # ref: https://cenleiding.github.io/神经网络ANN.html
    W1 = np.random.uniform(-np.sqrt(6 / (128 + 7)), np.sqrt(6 / (128 + 7)), size=(128, 7))
    W2 = np.random.uniform(-np.sqrt(6 / (64 + 128)), np.sqrt(6 / (64 + 128)), size=(64,128))
    W3 = np.random.uniform(-np.sqrt(6 / (64 + 2)), np.sqrt(6 / (64 + 2)), size=(2,64))

    for i in range(epochs):
        for j in range(len(train_data)):
            data = train_data[j]
            label = train_label[j]

            # modify data
            data = np.append(data, data[0] ** 2)
            data = np.append(data, data[1] ** 2)
            data = np.append(data, np.sin(data[1]))
            data = np.append(data, np.cos(data[1]))
            data = np.append(data, data[0] * data[1])
            
            X1 = data
            Y1 = np.dot(W1, X1)
            X2 = tanh(Y1)
            Y2 = np.dot(W2, X2)
            X3 = tanh(Y2)
            Y3 = np.dot(W3, X3)
            X4 = softmax(Y3)

            error = calc_err(X4, Y3, label)
            W3 -= learning_rate * np.outer(error, X3)
            error = np.dot(W3.T, error) * d_tanh(Y2)
            W2 -= learning_rate * np.outer(error, X2)
            error = np.dot(W2.T, error) * d_tanh(Y1)
            W1 -= learning_rate * np.outer(error, X1)

    prediction = []
    for data in test_data:
        data = np.append(data, data[0] ** 2)
        data = np.append(data, data[1] ** 2)
        data = np.append(data, np.sin(data[1]))
        data = np.append(data, np.cos(data[1]))
        data = np.append(data, data[0] * data[1])

        X1 = data
        Y1 = np.dot(W1, X1)
        X2 = tanh(Y1)
        Y2 = np.dot(W2, X2)
        X3 = tanh(Y2)
        Y3 = np.dot(W3, X3)
        X4 = softmax(Y3)
        prediction.append((np.argmax(X4)))

    np.savetxt('test_predictions.csv', prediction, delimiter='\n', fmt='%i')