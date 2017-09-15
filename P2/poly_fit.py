import numpy as np
import loadFittingDataP2 as loadData
import matplotlib.pyplot as plt
import math

def find_poly_fit(X, Y, M):
    '''
    X is list of data points
    Y is list of y values
    M is maximum degree of polynomial
    returns coefficients of poly of deg-M polynomial fitting points
    '''
    X_poly_basis = get_poly_basis(X, M)
    
    XTX_inv = np.linalg.inv(np.transpose(X_poly_basis).dot(X_poly_basis))
    XTY = np.transpose(X_poly_basis).dot(Y)
    poly = XTX_inv.dot(XTY)
    return poly

def get_poly_basis(X, M):
    '''
    X is a list
    M is the highest power of x in X we take
    returns [x^0, x^1, ..., x^M] for x in X
    '''
    basis = []
    for i in range(M + 1):
        basis.append(np.power(X, i))
    return np.transpose(np.vstack(basis))

def eval_poly(poly, x):
    '''
    poly is array of coefficients
    x is coordinate at which we wish to evaluate the polynomial
    returns evaluation of polynomial at x
    '''
    powers = []
    for i in range(len(poly)):
        powers.append(math.pow(x, i))
    powers = np.array(powers)
    return poly.dot(powers)

def eval_sse(poly, X, Y):
    '''
    poly is array of coefficients
    X and Y are data
    returns SSE
    '''
    y_hat = np.array(list(map(lambda a: eval_poly(poly, a), X)))
    dif = Y - y_hat
    return dif.dot(dif)

def gradient_sse(poly, X, Y):
    '''
    poly is array of coefficients representing a polynomial
    X and Y are data
    returns gradient of SSE wrt coefficients of poly
    '''
    X_poly_basis = get_poly_basis(X, M)
    XTY = np.transpose(X_poly_basis).dot(Y)
    XTXpoly = np.transpose(X_poly_basis).dot(X_poly_basis).dot(poly)
    return -2 * XTY + 2 * XTXpoly
    
def plot_poly_fit(X, Y, M):
    '''
    X is list of data points
    Y is list of y values
    M is maximum degree of polynomial
    plots graph of degree M fit
    '''
    poly = find_poly_fit(X, Y, M)
    x = np.linspace(0,1,100)
    y = list(map(lambda a: eval_poly(poly, a), x))
    plt.plot(x, y)
    plt.plot(X, Y, 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    
X, Y = loadData.getData(False)
M = 2
plot_poly_fit(X, Y, M)
coef = find_poly_fit(X, Y, M)
print("SSE", eval_sse(coef, X, Y))
print("GRADIENT SSE", gradient_sse(coef, X, Y))

plt.show()