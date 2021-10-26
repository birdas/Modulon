import math


def net_input(x, w, m):
    """
    :param x : array of feature vectors
    :param w : array of weights values
    :param m : array of modulus values
    """
    return [sum(w[j] * i[j] for j in range(len(x[i]))) % m[i] for i in range(len(x))]


def sigmoid(z):
    sig = []
    for i in z:
        if i == float('inf'):
            sig.append(1.0)
        elif i == float('-inf'):
            sig.append(0.0)
        else: 
            sig.append(1 / (1 + math.exp(-i))) 
    return sig


def logr_predict_proba(x, w):
    return sigmoid(net_input(x, w))


def logr_predict(x, w):
    pred = []
    for i in logr_predict_proba(x, w):
        if i >= 0.5:
            pred.append(1)
        else:
            pred.append(0)
    return pred


def logr_cost(x, y, w):
    m = len(x)
    h = logr_predict_proba(x, w)
    sum_ = sum([(y[i] * math.log(h[i])) + (1 - y[i]) * math.log(1 - h[i]) for i in range(m)])
    return sum_ / (-1 * m)


def logr_gradient(x, y, w):
    m = len(x)
    phi = logr_predict_proba(x, w)
    grad = [0 for i in range(len(w))]
    for i in range(len(grad)):
        grad[i] += sum([(y[j] - phi[j]) * (-1 * x[j][i]) for j in range(m)])
        grad[i] /= m
    return grad


def logr_gradient_descent(x, y, w_init, m_init, eta, n_iter):
    w = w_init
    m = m_init
    for _ in range(n_iter):
        print(logr_cost(x, y, w))
        gradient = logr_gradient(x, y, w)
        for i in range(len(w)):
            w[i] -= eta * gradient[i]
    return w
