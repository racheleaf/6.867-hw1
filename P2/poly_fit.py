import numpy as np
import loadFittingDataP2 as loadData
import matplotlib.pyplot as plt
import math

def find_fit(X_basis, Y):
    '''
    X_basis is full basis
    Y is list of y values
    returns linear fit of Y for X_basis
    '''
    XTX_inv = np.linalg.inv(np.transpose(X_basis).dot(X_basis))
    XTY = np.transpose(X_basis).dot(Y)
    return XTX_inv.dot(XTY)

def find_cos_fit(X, Y, M):
    '''
    X, Y lists
    returns coefficients of sum of cosine fit
    '''
    X_cos_basis = get_cos_basis(X, M)
    return find_fit(X_cos_basis, Y)

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

def get_cos_basis(X, M):
    '''
    X is list
    returns [cos(0pix), cos(1pix), ..., cos(Mpix)] for x in X
    '''
    basis = []
    for i in range(M + 1):
        basis.append(np.cos(i * math.pi * X))
    return np.transpose(np.vstack(basis))

def eval_poly(x, poly):
    '''
    x is point at which we wish to evaluate the polynomial
    poly is array of coefficients
    returns evaluation of polynomial at x
    '''
    powers = []
    for i in range(len(poly)):
        powers.append(math.pow(x, i))
    powers = np.array(powers)
    return poly.dot(powers)

def eval_cos_func(x, coeff):
    '''
    x is point at which we wish to evaluate sum of cos
    coeff is array of coefficients
    returns evaluation at x
    '''
    cos_list = []
    for i in range(len(coeff)):
        cos_list.append(math.cos(i * math.pi * x))
    cos_list = np.array(cos_list)
    return coeff.dot(cos_list)

def eval_sse(X, Y, f):
    '''
    X and Y are data
    returns SSE
    f is function taking single x point to y
    '''
    y_hat = np.array(list(map(f, X)))
    dif = Y - y_hat
    return dif.dot(dif)

def gradient_sse_poly(X, Y, poly):
    '''
    X and Y are data
    poly is array of coefficients representing a polynomial
    returns gradient of SSE wrt coefficients of poly
    '''
    X_poly_basis = get_poly_basis(X, M)
    XTY = np.transpose(X_poly_basis).dot(Y)
    XTXpoly = np.transpose(X_poly_basis).dot(X_poly_basis).dot(poly)
    return -2 * XTY + 2 * XTXpoly

def plot(X, Y, f):
    '''
    X is list of data points
    Y is list of y values
    f is function we want to plot
    '''
    x = np.linspace(0,1,100)
    y = list(map(f, x))
    plt.plot(x, y)
    plt.plot(X, Y, 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    
def plot_poly_fit(X, Y, M):
    '''
    X is list of data points
    Y is list of y values
    M is maximum degree of polynomial
    plots graph of degree M fit
    '''
    X_poly_basis = get_poly_basis(X, M)
    poly = find_fit(X_poly_basis, Y)
    f = lambda a: eval_poly(a, poly)
    plot(X, Y, f)
    
def plot_cos_fit(X, Y, M):
    '''
    X is list of data points
    Y is list of y values
    M is maximum value of cos(m * pi * x) used in basis
    plots graph with a0 cos(0pix) + a1 cos(1pix) + ... + aM cos(Mpix) fit
    '''
    X_cos_basis = get_cos_basis(X, M)
    coeff = find_fit(X_cos_basis, Y)
    f = lambda a: eval_cos_func(a, coeff)
    plot(X, Y, f)

def run_things(): 
    X, Y = loadData.getData(False)
    M = 2
    plot_poly_fit(X, Y, M)
    plot_cos_fit(X, Y, M)
    X_poly_basis = get_poly_basis(X, M)
    poly = find_fit(X_poly_basis, Y)
    eval_poly_at = lambda a: eval_poly(a, poly)
    print("POLY SSE", eval_sse(X, Y, eval_poly_at))
    print("POLY GRADIENT SSE", gradient_sse_poly(X, Y, poly))
    print("COSINE FIT", find_cos_fit(X, Y, M))

    plt.show()

# run_things()