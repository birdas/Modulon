import math


def net_input(x, w, m):
    """
    Net input method. 
    :param x : array of feature vectors
    :param w : array of weights values
    :param m : array of modulus values
    """
    return [sum(w[j] * x[i][j] for j in range(len(x[i]))) % m[i] for i in range(len(x))]


def sigmoid(z):
    """
    Sigmoid activation function.
    :param z : input to the activation funciton
    """
    sig = []
    for i in z:
        if i == float('inf'):
            sig.append(1.0)
        elif i == float('-inf'):
            sig.append(0.0)
        else: 
            sig.append(1 / (1 + math.exp(-i))) 
    return sig


def logr_predict_proba(x, w, m):
    """
    Predict probability method.
    :param x : array of feature vectors
    :param w : array of weight values
    :param m : array of modulus values
    """
    return sigmoid(net_input(x, w, m))


def logr_predict(x, w, m):
    """
    Prdeict method.
    :param x : array of feature vectors
    :param w : array of weight values
    :param m : array of modulus values
    """
    pred = []
    for i in logr_predict_proba(x, w, m):
        if i >= 0.5:
            pred.append(1)
        else:
            pred.append(0)
    return pred


def logr_cost(x, y, w, m):
    """
    Log loss cost function method.
    :param x : array of feature vectors
    :param y : array of expected outputs
    :param w : array of weight values
    :param m : array of modulus values
    """
    l = len(x)
    h = logr_predict_proba(x, w, m)
    sum_ = sum([(y[i] * math.log(h[i])) + (1 - y[i]) * math.log(1 - h[i]) for i in range(l)])
    return sum_ / (-1 * l)


def logr_gradient(x, y, w, m):
    """
    Gradient computation function.
    :param x : array of feature vectors
    :param y : array of expected outputs
    :param w : array of weight values
    :param m : array of modulus values
    """
    l = len(x)
    phi = logr_predict_proba(x, w, m)
    grad = [0 for i in range(len(w))]
    for i in range(len(grad)):
        grad[i] += sum([(y[j] - phi[j]) * (-1 * x[j][i]) for j in range(l)])
        grad[i] /= l
    return grad


def logr_gradient_descent(x, y, w_init, m_init, eta, n_iter):
    """
    Gradient descent optimization method.
    :param x : array of feature vectors
    :param y : array of expected outputs
    :param w_init : array of weight initialization values
    :param m_init : array of modulus initialization values
    :param eta : learning rate parameter
    :param n_iter : epochs paraameter
    """
    w = w_init
    m = m_init
    for _ in range(n_iter):
        print(logr_cost(x, y, w, m))
        gradient = logr_gradient(x, y, w, m)
        for i in range(len(w)):
            w[i] -= eta * gradient[i]
            m[i] = int(m[i] - (eta * gradient[i]))
    return w



print('Net Input:')
x = [[1, 2], [3, 4], [-1, 2.5]]
w = [2, 0.5]
m = [4, 1]
print(net_input(x, w, m))
print()
"""

print('Sigmoid:')
z = 0
print(sigmoid([z]))
print(sigmoid([float('inf')]))
print(sigmoid([float('-inf')]))
print(sigmoid([-1, 0, 1]))
print(sigmoid([-100, 100]))
print()


print('Predict Proba:')
x = [[7, 4, 8], [0, 0, 2], [7, 7, 4], [3, 0, 8]]
w = [-1, -2, 1]
print(logr_predict_proba(x, w))
x = [[3, 7, 8, 4, 1],
[3, 0, 7, 9, 5],
[8, 9, 3, 3, 7],
[4, 4, 9, 2, 6],
[4, 4, 5, 3, 6],
[7, 9, 0, 4, 9],
[1, 2, 6, 5, 1],
[6, 8, 8, 2, 8],
[2, 2, 1, 7, 3],
[1, 1, 3, 2, 6]]
w = [1, 1, -1, 1, -2]
print(logr_predict_proba(x, w))
print()


print('Log Predict:')
x = [[7, 4, 8], [0, 0, 2], [7, 7, 4], [3, 0, 8]]
w = [-1, -2, 1]
print(logr_predict(x, w))
x = [[3, 7, 8, 4, 1],
[3, 0, 7, 9, 5],
[8, 9, 3, 3, 7],
[4, 4, 9, 2, 6],
[4, 4, 5, 3, 6],
[7, 9, 0, 4, 9],
[1, 2, 6, 5, 1],
[6, 8, 8, 2, 8],
[2, 2, 1, 7, 3],
[1, 1, 3, 2, 6]]
w = [1, 1, -1, 1, -2]
print(logr_predict(x, w))
print()


print('Log cost:')
x = [[7, 4, 8], [0, 0, 2], [7, 7, 4], [3, 0, 8]]
y = [0, 1, 0, 1]
w = [-1, -2, 1]
print(logr_cost(x, y, w))
x = [[3, 7, 8, 4, 1],
[3, 0, 7, 9, 5],
[8, 9, 3, 3, 7],
[4, 4, 9, 2, 6],
[4, 4, 5, 3, 6],
[7, 9, 0, 4, 9],
[1, 2, 6, 5, 1],
[6, 8, 8, 2, 8],
[2, 2, 1, 7, 3],
[1, 1, 3, 2, 6]]
y = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
w = [1, 1, -1, 1, -2]
print(logr_cost(x, y, w))
print()


print('Gradient:')
x = [[7, 4, 8], [0, 0, 2], [7, 7, 4], [3, 0, 8]]
y = [0, 1, 0, 1]
w = [-1, -2, 1]
print(logr_gradient(x, y, w))
x = [[3, 7, 8, 4, 1],
[3, 0, 7, 9, 5],
[8, 9, 3, 3, 7],
[4, 4, 9, 2, 6],
[4, 4, 5, 3, 6],
[7, 9, 0, 4, 9],
[1, 2, 6, 5, 1],
[6, 8, 8, 2, 8],
[2, 2, 1, 7, 3],
[1, 1, 3, 2, 6]]
y = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
w = [1, 1, -1, 1, -2]
print(logr_gradient(x, y, w))
print()


print('Gradient Descent:')
x = [[7, 4, 8], [0, 0, 2], [7, 7, 4], [3, 0, 8]]
y = [0, 1, 0, 1]
w_init = [0, 0, 0]
eta = 0.3
n_iter = 10
print(logr_gradient_descent(x, y, w_init, eta, n_iter))
x = [[3, 7, 8, 4, 1],
[3, 0, 7, 9, 5],
[8, 9, 3, 3, 7],
[4, 4, 9, 2, 6],
[4, 4, 5, 3, 6],
[7, 9, 0, 4, 9],
[1, 2, 6, 5, 1],
[6, 8, 8, 2, 8],
[2, 2, 1, 7, 3],
[1, 1, 3, 2, 6]]
y = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
w_init = [0, 0, 0, 0, 0]
eta = 0.1
n_iter = 10
print(logr_gradient_descent(x, y, w_init, eta, n_iter))
n_iter = 1000
print(logr_gradient_descent(x, y, w_init, eta, n_iter))
"""
