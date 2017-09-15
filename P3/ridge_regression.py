import numpy as np
import loadFittingDataP2 as loadData
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('../P2')
import poly_fit

def ridge_regression(X, Y, M, l):
    '''
    X is a list
    M is the highest power of x in the basis
    l is value of lambda
    returns coefficients of deg M poly
    '''
    X_poly_basis = poly_fit.get_poly_basis(X, M)
    XTXlI_inv = np.linalg.inv(np.transpose(X_poly_basis).dot(X_poly_basis) + l * np.identity(M + 1))
    XTY = np.transpose(X_poly_basis).dot(Y)
    return XTXlI_inv.dot(XTY)
    
def plot_poly_fit(X, Y, M, l):
    '''
    X is list of data points
    Y is list of y values
    M is maximum degree of polynomial
    plots graph of degree M fit
    '''
    poly = ridge_regression(X, Y, M, l)
    f = lambda a: poly_fit.eval_poly(a, poly)
    poly_fit.plot(X, Y, f)

def run_things(): 
    # ridge regression for different values of lambda
    X, Y = loadData.getData(False)
    plot_poly_fit(X, Y, 2, 0.1)

run_things()
